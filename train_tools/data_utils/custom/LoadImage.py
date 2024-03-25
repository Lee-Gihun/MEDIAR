import numpy as np
import tifffile as tif
import skimage.io as io
from typing import Optional, Sequence, Union
from monai.config import DtypeLike, PathLike, KeysCollection
from monai.utils import ensure_tuple
from monai.data.utils import is_supported_format, optional_import, ensure_tuple_rep
from monai.data.image_reader import ImageReader, NumpyReader
from monai.transforms import LoadImage, LoadImaged
from monai.utils.enums import PostFix

DEFAULT_POST_FIX = PostFix.meta()
itk, has_itk = optional_import("itk", allow_namespace_pkg=True)

__all__ = [
    "CustomLoadImaged",
    "CustomLoadImageD",
    "CustomLoadImageDict",
    "CustomLoadImage",
]


class CustomLoadImage(LoadImage):
    """
    Load image file or files from provided path based on reader.
    If reader is not specified, this class automatically chooses readers
    based on the supported suffixes and in the following order:

        - User-specified reader at runtime when calling this loader.
        - User-specified reader in the constructor of `LoadImage`.
        - Readers from the last to the first in the registered list.
        - Current default readers: (nii, nii.gz -> NibabelReader), (png, jpg, bmp -> PILReader),
          (npz, npy -> NumpyReader), (nrrd -> NrrdReader), (DICOM file -> ITKReader).

    [!Caution] This overriding replaces the original ITK with Custom UnifiedITKReader.
    """

    def __init__(
        self,
        reader=None,
        image_only: bool = False,
        dtype: DtypeLike = np.float32,
        ensure_channel_first: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super(CustomLoadImage, self).__init__(
            reader, image_only, dtype, ensure_channel_first, *args, **kwargs
        )

        # Adding TIFFReader. Although ITK Reader supports ".tiff" files, sometimes fails to load images.
        self.readers = []
        self.register(UnifiedITKReader(*args, **kwargs))


class CustomLoadImaged(LoadImaged):
    """
    Dictionary-based wrapper of `CustomLoadImage`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys=False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super(CustomLoadImaged, self).__init__(
            keys,
            reader,
            dtype,
            meta_keys,
            meta_key_postfix,
            overwriting,
            image_only,
            ensure_channel_first,
            simple_keys,
            allow_missing_keys,
            *args,
            **kwargs,
        )

        # Assign CustomLoader
        self._loader = CustomLoadImage(
            reader, image_only, dtype, ensure_channel_first, *args, **kwargs
        )
        if not isinstance(meta_key_postfix, str):
            raise TypeError(
                f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}."
            )
        self.meta_keys = (
            ensure_tuple_rep(None, len(self.keys))
            if meta_keys is None
            else ensure_tuple(meta_keys)
        )
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


class UnifiedITKReader(NumpyReader):
    """
    Unified Reader to read ".tif" and ".tiff files".
    As the tifffile reads the images as numpy arrays, it inherits from the NumpyReader.
    """

    def __init__(
        self, channel_dim: Optional[int] = None, **kwargs,
    ):
        super(UnifiedITKReader, self).__init__(channel_dim=channel_dim, **kwargs)
        self.kwargs = kwargs
        self.channel_dim = channel_dim

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        """Verify whether the file format is supported by TIFF Reader."""

        suffixes: Sequence[str] = ["tif", "tiff", "png", "jpg", "bmp", "jpeg",]
        return has_itk or is_supported_format(filename, suffixes)

    def read(self, data: Union[Sequence[PathLike], PathLike], **kwargs):
        """Read Images from the file."""
        img_ = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)

        for name in filenames:
            name = f"{name}"

            if name.endswith(".tif") or name.endswith(".tiff"):
                _obj = tif.imread(name)
            else:
                try:
                    _obj = itk.imread(name, **kwargs_)
                    _obj = itk.array_view_from_image(_obj, keep_axes=False)
                except:
                    _obj = io.imread(name)

            if len(_obj.shape) == 2:
                _obj = np.repeat(np.expand_dims(_obj, axis=-1), 3, axis=-1)
            elif len(_obj.shape) == 3 and _obj.shape[-1] > 3:
                _obj = _obj[:, :, :3]
            else:
                pass

            img_.append(_obj)

        return img_ if len(filenames) > 1 else img_[0]


CustomLoadImageD = CustomLoadImageDict = CustomLoadImaged
