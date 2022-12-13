import itk
import torch
import numpy as np 
from monai.transforms import Affine
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

    To use: image[:, :, :] = remove_border(image)

    Args:
        image: The ITK image to be padded. 
        
    Returns:
        The padded array of data. 
    """
    return np.pad(image[1:-1, 1:-1, 1:-1], pad_width=1)

def itk_to_monai_affine(image, matrix, translation, center_of_rotation=None):
    """
    Converts an ITK affine matrix (3x3 matrix and translation vector) to a 
    MONAI affine matrix.
    
    Args:
        image: The ITK image object. This is used to extract the spacing and 
               direction information.
        matrix: The 3x3 ITK affine matrix.
        translation: The 3-element ITK affine translation vector.
        center_of_rotation: The center of rotation. If provided, the affine 
                            matrix will be adjusted to account for the difference
                            between the center of the image and the center of rotation.
        
    Returns:
        A 4x4 MONAI affine matrix.
    """

    # Create 4x4 affine matrix
    affine_matrix = torch.eye(4, dtype=torch.float64)
    affine_matrix[:3, :3] = torch.tensor(np.asarray(matrix, dtype=np.float64))
    affine_matrix[:3, 3] = torch.tensor(translation, dtype=torch.float64) 

    # Adjust offset when center of rotation is different from center of the image
    if center_of_rotation:
        offset = np.asarray(get_itk_image_center(image)) - np.asarray(center_of_rotation)
        offset_matrix = torch.eye(4, dtype=torch.float64)
        offset_matrix[:3, 3] = torch.tensor(offset, dtype=torch.float64)
        inverse_offset_matrix = torch.eye(4, dtype=torch.float64)
        inverse_offset_matrix[:3, 3] = -torch.tensor(offset, dtype=torch.float64)
        
        affine_matrix = inverse_offset_matrix @ affine_matrix @ offset_matrix

    # Adjust based on spacing. It is required because MONAI does not update the 
    # pixel array according to the spacing after a transformation. For example,
    # a rotation of 90deg for an image with different spacing along the two axis
    # will just rotate the image array by 90deg without also scaling accordingly.
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    spacing_matrix = torch.eye(4, dtype=torch.float64)
    inverse_spacing_matrix = torch.eye(4, dtype=torch.float64)
    for i, e in enumerate(spacing):
        spacing_matrix[i, i] = e
        inverse_spacing_matrix[i, i] = 1 / e
    
    affine_matrix = inverse_spacing_matrix @ affine_matrix @ spacing_matrix 

    # Adjust direction
    direction = itk.array_from_matrix(image.GetDirection())
    direction_matrix = torch.eye(4, dtype=torch.float64) 
    direction_matrix[:3, :3] = torch.tensor(direction, dtype=torch.float64)
    inverse_direction = itk.array_from_matrix(image.GetInverseDirection())
    inverse_direction_matrix = torch.eye(4, dtype=torch.float64) 
    inverse_direction_matrix[:3, :3] = torch.tensor(inverse_direction, dtype=torch.float64)

    affine_matrix = inverse_direction_matrix @ affine_matrix @ direction_matrix

    return affine_matrix

    
def get_itk_image_center(image):
    """
    Calculates the center of the ITK image based on its origin, size, and spacing.
    This center is equivalent to the implicit image center that MONAI uses.

    Args:
        image: The ITK image.

    Returns:
        The center of the image as a list of coordinates.
    """
    image_size = np.asarray(image.GetLargestPossibleRegion().GetSize(), np.float32)
    spacing = np.asarray(image.GetSpacing())
    origin = np.asarray(image.GetOrigin())
    center = image.GetDirection() @ ((image_size / 2 - 0.5) * spacing) + origin
    
    return center.tolist()


def create_itk_affine_from_parameters(image, translation=None, rotation=None, 
                                      scale=None, shear=None, 
                                      center_of_rotation=None):
    """
    Creates an affine transformation for an ITK image based on the provided parameters.

    Args:
        image: The ITK image.
        translation: The translation (shift) to apply to the image.
        rotation: The rotation to apply to the image, specified as angles in radians 
                around the x, y, and z axes.
        scale: The scaling factor to apply to the image.
        shear: The shear to apply to the image.
        center_of_rotation: The center of rotation for the image. If not specified, 
                            the center of the image is used.

    Returns:
        A tuple containing the affine transformation matrix and the translation vector.
    """
    itk_transform = itk.AffineTransform[itk.D, image.ndim].New()

    # Set center
    if center_of_rotation:
        itk_transform.SetCenter(center_of_rotation)
    else:
        itk_transform.SetCenter(get_itk_image_center(image))

    # Set parameters
    if rotation:
        for i, angle_in_rads in enumerate(rotation):
            if angle_in_rads != 0:
                axis = [0, 0, 0]
                axis[i] = 1
                itk_transform.Rotate3D(axis, angle_in_rads)

    if scale:
        itk_transform.Scale(scale)

    if shear:
        itk_transform.Shear(*shear)

    if translation:
        itk_transform.Translate(translation)

    matrix = itk_transform.GetMatrix()

    return matrix, translation


def transform_affinely_with_transformix(image, translation, matrix, center_of_rotation=None):
    sz = tuple([str(e) for e in image.GetLargestPossibleRegion().GetSize()])
    spacing = tuple([str(e) for e in image.GetSpacing()])
    direction = tuple([str(e) for e in np.asarray(image.GetDirection()).flatten()])
    origin = tuple([str(e) for e in np.asarray(image.GetOrigin()).flatten()])
    index = tuple([str(e) for e in np.zeros(image.ndim, dtype=np.int32)])

    parameter_map = {
                    "Direction": direction,
                    "Index": index, 
                    "Origin": origin, 
                    "Size": sz,
                    "Spacing": spacing,
                    "ResampleInterpolator": ("FinalLinearInterpolator", )
                    }

    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterMap(parameter_map) 

    # Transform
    itk_transform = itk.AffineTransform[itk.D, image.ndim].New()

    if center_of_rotation:
        itk_transform.SetCenter(center_of_rotation)
    else:
        itk_transform.SetCenter(get_itk_image_center(image))

    itk_transform.SetMatrix(matrix)
    itk_transform.Translate(translation)

    # Transformix
    transformix_filter = itk.TransformixFilter[type(image)].New()
    transformix_filter.SetMovingImage(image)
    transformix_filter.SetTransformParameterObject(parameter_object)
    transformix_filter.SetTransform(itk_transform)
    transformix_filter.Update()
    output_image = transformix_filter.GetOutput()

    return np.asarray(output_image, dtype=np.float32) 


def transform_affinely_with_itk(image, matrix, translation, center_of_rotation=None):
    # Translation transform
    itk_transform = itk.AffineTransform[itk.D, image.ndim].New()

    # Set center
    if center_of_rotation:
        itk_transform.SetCenter(center_of_rotation)
    else:
        itk_transform.SetCenter(get_itk_image_center(image))

    # Set matrix and translation
    itk_transform.SetMatrix(matrix)
    itk_transform.Translate(translation)

    # Interpolator
    image = image.astype(itk.D)
    interpolator = itk.LinearInterpolateImageFunction.New(image)
    # interpolator = itk.NearestNeighborInterpolateImageFunction.New(image)

    # Resample with ITK
    resampler = itk.ResampleImageFilter.New(image)
    resampler.SetInterpolator(interpolator)
    resampler.SetTransform(itk_transform)
    resampler.SetOutputParametersFromImage(image)
    resampler.Update()
    output_image = resampler.GetOutput()

    return np.asarray(output_image, dtype=np.float32)


def transform_affinely_with_monai(metatensor, affine_matrix):
    # monai_transform = Affine(translate_params=translation, mode=1, padding_mode="constant", 
                              # dtype=torch.float64) # numpy/cupy backend

    monai_transform = Affine(affine=affine_matrix, padding_mode="zeros", dtype=torch.float64)

    # output_tensor, output_affine = monai_transform(metatensor, mode='nearest')
    output_tensor, output_affine = monai_transform(metatensor, mode='bilinear')

    return metatensor_to_array(output_tensor)

