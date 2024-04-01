"""
Copyright © 2022 Howard Hughes Medical Institute, 
Authored by Carsen Stringer and Marius Pachitariu.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, 
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, 
   this list of conditions and the following disclaimer in the documentation 
   and/or other materials provided with the distribution.

3. Neither the name of HHMI nor the names of its contributors may be used to 
   endorse or promote products derived from this software without specific 
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.

--------------------------------------------------------------------------
MEDIAR Prediction uses CellPose's Gradient Flow Tracking.

This code is adapted from the following codes:
[1] https://github.com/MouseLand/cellpose/blob/main/cellpose/utils.py
[2] https://github.com/MouseLand/cellpose/blob/main/cellpose/dynamics.py
[3] https://github.com/MouseLand/cellpose/blob/main/cellpose/metrics.py
"""

import torch
from torch.nn.functional import grid_sample
import numpy as np
import fastremap

from skimage import morphology
from scipy.ndimage import mean, find_objects
from scipy.ndimage.filters import maximum_filter1d

torch_GPU = torch.device("cuda")
torch_CPU = torch.device("cpu")


def labels_to_flows(labels, use_gpu=False, device=None, redo_flows=False):
    """
    Convert labels (list of masks or flows) to flows for training model
    """

    # Labels b x 1 x h x w
    labels = labels.cpu().numpy().astype(np.int16)
    nimg = len(labels)

    if labels[0].ndim < 3:
        labels = [labels[n][np.newaxis, :, :] for n in range(nimg)]

    # Flows need to be recomputed
    if labels[0].shape[0] == 1 or labels[0].ndim < 3 or redo_flows:
        # compute flows; labels are fixed here to be unique, so they need to be passed back
        # make sure labels are unique!
        labels = [fastremap.renumber(label, in_place=True)[0] for label in labels]
        veci = [
            masks_to_flows(labels[n][0], use_gpu=use_gpu, device=device)
            for n in range(nimg)
        ]

        # concatenate labels, distance transform, vector flows, heat (boundary and mask are computed in augmentations)
        flows = [
            np.concatenate((labels[n], labels[n] > 0.5, veci[n]), axis=0).astype(
                np.float32
            )
            for n in range(nimg)
        ]

    return np.array(flows)


def compute_masks(
    dP,
    cellprob,
    p=None,
    niter=200,
    cellprob_threshold=0.4,
    flow_threshold=0.4,
    interp=True,
    resize=None,
    use_gpu=False,
    device=None,
):
    """compute masks using dynamics from dP, cellprob, and boundary"""

    cp_mask = cellprob > cellprob_threshold
    cp_mask = morphology.remove_small_holes(cp_mask, area_threshold=16)
    cp_mask = morphology.remove_small_objects(cp_mask, min_size=16)

    if np.any(cp_mask):  # mask at this point is a cell cluster binary map, not labels
        # follow flows
        if p is None:
            p, inds = follow_flows(
                dP * cp_mask / 5.0,
                niter=niter,
                interp=interp,
                use_gpu=use_gpu,
                device=device,
            )
            if inds is None:
                shape = resize if resize is not None else cellprob.shape
                mask = np.zeros(shape, np.uint16)
                p = np.zeros((len(shape), *shape), np.uint16)
                return mask, p

        # calculate masks
        mask = get_masks(p, iscell=cp_mask)
        
        # flow thresholding factored out of get_masks
        shape0 = p.shape[1:]
        if mask.max() > 0 and flow_threshold is not None and flow_threshold > 0:
            # make sure labels are unique at output of get_masks
            mask = remove_bad_flow_masks(
                mask, dP, threshold=flow_threshold, use_gpu=use_gpu, device=device
            )
        else:  # nothing to compute, just make it compatible
            shape = resize if resize is not None else cellprob.shape
            mask = np.zeros(shape, np.uint16)
            p = np.zeros((len(shape), *shape), np.uint16)

    return mask, p


def _extend_centers_gpu(
    neighbors, centers, isneighbor, Ly, Lx, n_iter=200, device=torch.device("cuda")
):
    if device is not None:
        device = device
    nimg = neighbors.shape[0] // 9
    pt = torch.from_numpy(neighbors).to(device)

    T = torch.zeros((nimg, Ly, Lx), dtype=torch.double, device=device)
    meds = torch.from_numpy(centers.astype(int)).to(device).long()
    isneigh = torch.from_numpy(isneighbor).to(device)
    for i in range(n_iter):
        T[:, meds[:, 0], meds[:, 1]] += 1
        Tneigh = T[:, pt[:, :, 0], pt[:, :, 1]]
        Tneigh *= isneigh
        T[:, pt[0, :, 0], pt[0, :, 1]] = Tneigh.mean(axis=1)
    del meds, isneigh, Tneigh
    T = torch.log(1.0 + T)
    # gradient positions
    grads = T[:, pt[[2, 1, 4, 3], :, 0], pt[[2, 1, 4, 3], :, 1]]
    del pt
    dy = grads[:, 0] - grads[:, 1]
    dx = grads[:, 2] - grads[:, 3]
    del grads
    mu_torch = np.stack((dy.cpu().squeeze(), dx.cpu().squeeze()), axis=-2)
    return mu_torch


def diameters(masks):
    _, counts = np.unique(np.int32(masks), return_counts=True)
    counts = counts[1:]
    md = np.median(counts ** 0.5)
    if np.isnan(md):
        md = 0
    md /= (np.pi ** 0.5) / 2
    return md, counts ** 0.5


def masks_to_flows_gpu(masks, device=None):
    if device is None:
        device = torch.device("cuda")

    Ly0, Lx0 = masks.shape
    Ly, Lx = Ly0 + 2, Lx0 + 2

    masks_padded = np.zeros((Ly, Lx), np.int64)
    masks_padded[1:-1, 1:-1] = masks

    # get mask pixel neighbors
    y, x = np.nonzero(masks_padded)
    neighborsY = np.stack((y, y - 1, y + 1, y, y, y - 1, y - 1, y + 1, y + 1), axis=0)
    neighborsX = np.stack((x, x, x, x - 1, x + 1, x - 1, x + 1, x - 1, x + 1), axis=0)
    neighbors = np.stack((neighborsY, neighborsX), axis=-1)

    # get mask centers
    slices = find_objects(masks)

    centers = np.zeros((masks.max(), 2), "int")
    for i, si in enumerate(slices):
        if si is not None:
            sr, sc = si

            ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
            yi, xi = np.nonzero(masks[sr, sc] == (i + 1))
            yi = yi.astype(np.int32) + 1  # add padding
            xi = xi.astype(np.int32) + 1  # add padding
            ymed = np.median(yi)
            xmed = np.median(xi)
            imin = np.argmin((xi - xmed) ** 2 + (yi - ymed) ** 2)
            xmed = xi[imin]
            ymed = yi[imin]
            centers[i, 0] = ymed + sr.start
            centers[i, 1] = xmed + sc.start

    # get neighbor validator (not all neighbors are in same mask)
    neighbor_masks = masks_padded[neighbors[:, :, 0], neighbors[:, :, 1]]
    isneighbor = neighbor_masks == neighbor_masks[0]
    ext = np.array(
        [[sr.stop - sr.start + 1, sc.stop - sc.start + 1] for sr, sc in slices]
    )
    n_iter = 2 * (ext.sum(axis=1)).max()
    # run diffusion
    mu = _extend_centers_gpu(
        neighbors, centers, isneighbor, Ly, Lx, n_iter=n_iter, device=device
    )

    # normalize
    mu /= 1e-20 + (mu ** 2).sum(axis=0) ** 0.5

    # put into original image
    mu0 = np.zeros((2, Ly0, Lx0))
    mu0[:, y - 1, x - 1] = mu
    mu_c = np.zeros_like(mu0)
    return mu0, mu_c


def masks_to_flows(masks, use_gpu=False, device=None):
    if masks.max() == 0 or (masks != 0).sum() == 1:
        # dynamics_logger.warning('empty masks!')
        return np.zeros((2, *masks.shape), "float32")

    if use_gpu:
        if use_gpu and device is None:
            device = torch_GPU
        elif device is None:
            device = torch_CPU
        masks_to_flows_device = masks_to_flows_gpu

    if masks.ndim == 3:
        Lz, Ly, Lx = masks.shape
        mu = np.zeros((3, Lz, Ly, Lx), np.float32)
        for z in range(Lz):
            mu0 = masks_to_flows_device(masks[z], device=device)[0]
            mu[[1, 2], z] += mu0
        for y in range(Ly):
            mu0 = masks_to_flows_device(masks[:, y], device=device)[0]
            mu[[0, 2], :, y] += mu0
        for x in range(Lx):
            mu0 = masks_to_flows_device(masks[:, :, x], device=device)[0]
            mu[[0, 1], :, :, x] += mu0
        return mu
    elif masks.ndim == 2:
        mu, mu_c = masks_to_flows_device(masks, device=device)
        return mu

    else:
        raise ValueError("masks_to_flows only takes 2D or 3D arrays")


def steps2D_interp(p, dP, niter, use_gpu=False, device=None):
    shape = dP.shape[1:]
    if use_gpu:
        if device is None:
            device = torch_GPU
        shape = (
            np.array(shape)[[1, 0]].astype("float") - 1
        )  # Y and X dimensions (dP is 2.Ly.Lx), flipped X-1, Y-1
        pt = (
            torch.from_numpy(p[[1, 0]].T).float().to(device).unsqueeze(0).unsqueeze(0)
        )  # p is n_points by 2, so pt is [1 1 2 n_points]
        im = (
            torch.from_numpy(dP[[1, 0]]).float().to(device).unsqueeze(0)
        )  # covert flow numpy array to tensor on GPU, add dimension
        # normalize pt between  0 and  1, normalize the flow
        for k in range(2):
            im[:, k, :, :] *= 2.0 / shape[k]
            pt[:, :, :, k] /= shape[k]

        # normalize to between -1 and 1
        pt = pt * 2 - 1

        # here is where the stepping happens
        for t in range(niter):
            # align_corners default is False, just added to suppress warning
            dPt = grid_sample(im, pt, align_corners=False)

            for k in range(2):  # clamp the final pixel locations
                pt[:, :, :, k] = torch.clamp(
                    pt[:, :, :, k] + dPt[:, k, :, :], -1.0, 1.0
                )

        # undo the normalization from before, reverse order of operations
        pt = (pt + 1) * 0.5
        for k in range(2):
            pt[:, :, :, k] *= shape[k]

        p = pt[:, :, :, [1, 0]].cpu().numpy().squeeze().T
        return p

    else:
        assert print("ho")


def follow_flows(dP, mask=None, niter=200, interp=True, use_gpu=True, device=None):
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.uint32(niter)

    p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    p = np.array(p).astype(np.float32)

    inds = np.array(np.nonzero(np.abs(dP[0]) > 1e-3)).astype(np.int32).T

    if inds.ndim < 2 or inds.shape[0] < 5:
        return p, None

    if not interp:
        assert print("woo")

    else:
        p_interp = steps2D_interp(
            p[:, inds[:, 0], inds[:, 1]], dP, niter, use_gpu=use_gpu, device=device
        )
        p[:, inds[:, 0], inds[:, 1]] = p_interp

    return p, inds


def flow_error(maski, dP_net, use_gpu=False, device=None):
    if dP_net.shape[1:] != maski.shape:
        print("ERROR: net flow is not same size as predicted masks")
        return

    # flows predicted from estimated masks
    dP_masks = masks_to_flows(maski, use_gpu=use_gpu, device=device)
    # difference between predicted flows vs mask flows
    flow_errors = np.zeros(maski.max())
    for i in range(dP_masks.shape[0]):
        flow_errors += mean(
            (dP_masks[i] - dP_net[i] / 5.0) ** 2,
            maski,
            index=np.arange(1, maski.max() + 1),
        )

    return flow_errors, dP_masks


def remove_bad_flow_masks(masks, flows, threshold=0.4, use_gpu=False, device=None):
    merrors, _ = flow_error(masks, flows, use_gpu, device)
    badi = 1 + (merrors > threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks


def get_masks(p, iscell=None, rpad=20):
    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = len(p)

    for i in range(dims):
        pflows.append(p[i].flatten().astype("int32"))
        edges.append(np.arange(-0.5 - rpad, shape0[i] + 0.5 + rpad, 1))

    h, _ = np.histogramdd(tuple(pflows), bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = maximum_filter1d(hmax, 5, axis=i)

    seeds = np.nonzero(np.logical_and(h - hmax > -1e-6, h > 10))
    Nmax = h[seeds]
    isort = np.argsort(Nmax)[::-1]
    for s in seeds:
        s = s[isort]

    pix = list(np.array(seeds).T)

    shape = h.shape
    if dims == 3:
        expand = np.nonzero(np.ones((3, 3, 3)))
    else:
        expand = np.nonzero(np.ones((3, 3)))
    for e in expand:
        e = np.expand_dims(e, 1)

    for iter in range(5):
        for k in range(len(pix)):
            if iter == 0:
                pix[k] = list(pix[k])
            newpix = []
            iin = []
            for i, e in enumerate(expand):
                epix = e[:, np.newaxis] + np.expand_dims(pix[k][i], 0) - 1
                epix = epix.flatten()
                iin.append(np.logical_and(epix >= 0, epix < shape[i]))
                newpix.append(epix)
            iin = np.all(tuple(iin), axis=0)
            for p in newpix:
                p = p[iin]
            newpix = tuple(newpix)
            igood = h[newpix] > 2
            for i in range(dims):
                pix[k][i] = newpix[i][igood]
            if iter == 4:
                pix[k] = tuple(pix[k])

    M = np.zeros(h.shape, np.uint32)
    for k in range(len(pix)):
        M[pix[k]] = 1 + k

    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    M0 = M[tuple(pflows)]

    # remove big masks
    uniq, counts = fastremap.unique(M0, return_counts=True)
    big = np.prod(shape0) * 0.9
    bigc = uniq[counts > big]
    if len(bigc) > 0 and (len(bigc) > 1 or bigc[0] != 0):
        M0 = fastremap.mask(M0, bigc)
    fastremap.renumber(M0, in_place=True)  # convenient to guarantee non-skipped labels
    M0 = np.reshape(M0, shape0)
    return M0
