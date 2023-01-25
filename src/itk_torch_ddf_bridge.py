import itk
import torch
import monai
import numpy as np
import matplotlib.pyplot as plt

def monai_to_itk_ddf(image, ddf):
    """
    converting the dense displacement field from the MONAI space to the ITK 
    Args:
        image: itk image of array shape 2D: (H, W) or 3D: (D, H, W)
        ddf: numpy array of shape 2D: (2, H, W) or 3D: (3, D, H, W)
    Returns:
        displacement_field: itk image of the corresponding displacement field
        
    """
    # 3, D, H, W -> D, H, W, 3
    ndim = image.ndim
    ddf = ddf.transpose(tuple(list(range(1, ndim+1)) + [0]))
    # x, y, z -> z, x, y
    ddf = ddf[..., ::-1]

    # Correct for spacing
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64) 
    ddf *= np.array(spacing, ndmin=ndim+1) 

    # Correct for direction
    direction = np.asarray(image.GetDirection(), dtype=np.float64)
    ddf = np.einsum('ij,...j->...i', direction, ddf, dtype=np.float64).astype(np.float32)

    # initialise displacement field - 
    vector_component_type = itk.F
    vector_pixel_type = itk.Vector[vector_component_type, ndim]
    displacement_field_type = itk.Image[vector_pixel_type, ndim]
    displacement_field = itk.GetImageFromArray(ddf, ttype=displacement_field_type)

    # Set image metadata 
    displacement_field.SetSpacing(image.GetSpacing())
    displacement_field.SetOrigin(image.GetOrigin())
    displacement_field.SetDirection(image.GetDirection())

    return displacement_field


def itk_warp(image, ddf):
    """
    warping with python itk
    Args:
        image: itk image of array shape 2D: (H, W) or 3D: (D, H, W)
        ddf: numpy array of shape 2D: (2, H, W) or 3D: (3, D, H, W)
    Returns:
        warped_image: numpy array of shape (H, W) or (D, H, W)
    """
    # MONAI->ITK ddf
    displacement_field = monai_to_itk_ddf(image, ddf)

    # Resample using the ddf 
    interpolator = itk.LinearInterpolateImageFunction.New(image)
    warped_image = itk.warp_image_filter(image,
                                       interpolator=interpolator,
                                       displacement_field=displacement_field,
                                       output_parameters_from_image=image)

    return np.asarray(warped_image)
    

def monai_wrap(image_tensor, ddf_tensor):
    """
    warping with MONAI
    Args:
        image_tensor: torch tensor of shape 2D: (1, 1, H, W) and 3D: (1, 1, D, H, W)
        ddf_tensor: torch tensor of shape 2D: (1, 2, H, W) and 3D: (1, 3, D, H, W)
    Returns:
        warped_image: numpy array of shape (H, W) or (D, H, W)
    """
    warp = monai.networks.blocks.Warp(mode='bilinear', padding_mode="zeros")
    warped_image = warp(image_tensor.to(torch.float64), ddf_tensor.to(torch.float64))
    
    return warped_image.to(torch.float32).squeeze().numpy()



