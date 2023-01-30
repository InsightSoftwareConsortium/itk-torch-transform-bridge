import itk
import torch
import numpy as np
from monai.data import ITKReader
from monai.data.meta_tensor import MetaTensor
from monai.transforms import EnsureChannelFirst
from monai.utils import convert_to_dst_type


def metatensor_to_array(metatensor):
    metatensor = metatensor.squeeze()
    metatensor = metatensor.permute(*torch.arange(metatensor.ndim - 1, -1, -1))

    return metatensor.get_array()


def image_to_metatensor(image):
    """
    Converts an ITK image to a MetaTensor object.
    
    Args:
        image: The ITK image to be converted. 
        
    Returns:
        A MetaTensor object containing the array data and metadata.
    """
    reader = ITKReader(affine_lps_to_ras=False)
    image_array, meta_data = reader.get_data(image)
    image_array = convert_to_dst_type(image_array, dst=image_array, dtype=itk.D)[0]
    metatensor = MetaTensor.ensure_torch_and_prune_meta(image_array, meta_data)
    metatensor = EnsureChannelFirst()(metatensor)

    return metatensor



def remove_border(image):
    """
    MONAI seems to have different behavior in the borders of the image than ITK.
    This helper function sets the border of the ITK image as 0 (padding but keeping
    the same image size) in order to allow numerical comparison between the 
    result from resampling with ITK/Elastix and resampling with MONAI.
    To use: image[:] = remove_border(image)
    Args:
        image: The ITK image to be padded. 
        
    Returns:
        The padded array of data. 
    """
    return np.pad(image[1:-1, 1:-1, 1:-1] if image.ndim==3 else image[1:-1, 1:-1],
                                              pad_width=1)
