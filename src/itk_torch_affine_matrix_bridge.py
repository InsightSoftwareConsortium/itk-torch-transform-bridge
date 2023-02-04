import copy
import itk
import torch
import numpy as np 
from monai.transforms import Affine
from monai.data import ITKReader
from monai.data.meta_tensor import MetaTensor
from monai.transforms import EnsureChannelFirst
from monai.utils import convert_to_dst_type

def assert_itk_regions_match_array(image):
    # Note: Make it more compact? Also, are there redundant checks?
    largest_region = image.GetLargestPossibleRegion()
    buffered_region = image.GetBufferedRegion()
    requested_region = image.GetRequestedRegion()

    largest_region_size = np.array(largest_region.GetSize())
    buffered_region_size = np.array(buffered_region.GetSize()) 
    requested_region_size = np.array(requested_region.GetSize()) 
    array_size = np.array(image.shape)[::-1]

    largest_region_index = np.array(largest_region.GetIndex())
    buffered_region_index = np.array(buffered_region.GetIndex())
    requested_region_index = np.array(requested_region.GetIndex())

    indices_are_zeros = np.all(largest_region_index==0) and \
                        np.all(buffered_region_index==0) and \
                        np.all(requested_region_index==0)

    sizes_match = np.array_equal(array_size, largest_region_size) and \
                  np.array_equal(largest_region_size, buffered_region_size) and \
                  np.array_equal(buffered_region_size, requested_region_size)

    assert indices_are_zeros, "ITK-MONAI bridge: non-zero ITK region indices encountered"
    assert sizes_match, "ITK-MONAI bridge: ITK regions should be of the same shape"


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
            
def compute_offset_matrix(image, center_of_rotation):
    ndim = image.ndim
    offset = np.asarray(get_itk_image_center(image)) - np.asarray(center_of_rotation)
    offset_matrix = torch.eye(ndim+1, dtype=torch.float64)
    offset_matrix[:ndim, ndim] = torch.tensor(offset, dtype=torch.float64)
    inverse_offset_matrix = torch.eye(ndim+1, dtype=torch.float64)
    inverse_offset_matrix[:ndim, ndim] = -torch.tensor(offset, dtype=torch.float64)

    return offset_matrix, inverse_offset_matrix

def compute_spacing_matrix(image):
    ndim = image.ndim
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    spacing_matrix = torch.eye(ndim+1, dtype=torch.float64)
    inverse_spacing_matrix = torch.eye(ndim+1, dtype=torch.float64)
    for i, e in enumerate(spacing):
        spacing_matrix[i, i] = e
        inverse_spacing_matrix[i, i] = 1 / e

    return spacing_matrix, inverse_spacing_matrix

def compute_direction_matrix(image):
    ndim = image.ndim
    direction = itk.array_from_matrix(image.GetDirection())
    direction_matrix = torch.eye(ndim+1, dtype=torch.float64) 
    direction_matrix[:ndim, :ndim] = torch.tensor(direction, dtype=torch.float64)
    inverse_direction = itk.array_from_matrix(image.GetInverseDirection())
    inverse_direction_matrix = torch.eye(ndim+1, dtype=torch.float64) 
    inverse_direction_matrix[:ndim, :ndim] = torch.tensor(inverse_direction, dtype=torch.float64)

    return direction_matrix, inverse_direction_matrix

def compute_reference_space_affine_matrix(image, ref_image): 
    ndim = ref_image.ndim

    # Spacing and direction as matrices
    spacing_matrix, inv_spacing_matrix = [m[:ndim, :ndim].numpy() for m in compute_spacing_matrix(image)]
    ref_spacing_matrix, ref_inv_spacing_matrix = [m[:ndim, :ndim].numpy() for m in compute_spacing_matrix(ref_image)]

    direction_matrix, inv_direction_matrix = [m[:ndim, :ndim].numpy() for m in compute_direction_matrix(image)]
    ref_direction_matrix, ref_inv_direction_matrix = [m[:ndim, :ndim].numpy() for m in compute_direction_matrix(ref_image)]

    # Matrix calculation
    matrix = ref_direction_matrix @ ref_spacing_matrix @ inv_spacing_matrix @ inv_direction_matrix 

    # Offset calculation
    pixel_offset = -1
    image_size = np.asarray(ref_image.GetLargestPossibleRegion().GetSize(), np.float32)
    translation =  (ref_direction_matrix @ ref_spacing_matrix - direction_matrix @ spacing_matrix) @ (image_size + pixel_offset) / 2 
    translation += np.asarray(ref_image.GetOrigin()) -  np.asarray(image.GetOrigin())

    # Convert matrix ITK matrix and translation to MONAI affine matrix
    ref_affine_matrix = itk_to_monai_affine(image, matrix=matrix, translation=translation)

    return ref_affine_matrix 


def itk_to_monai_affine(image, matrix, translation, center_of_rotation=None, reference_image=None):
    """
    Converts an ITK affine matrix (2x2 for 2D or 3x3 for 3D matrix and translation
    vector) to a MONAI affine matrix.
    
    Args:
        image: The ITK image object. This is used to extract the spacing and 
               direction information.
        matrix: The 2x2 or 3x3 ITK affine matrix.
        translation: The 2-element or 3-element ITK affine translation vector.
        center_of_rotation: The center of rotation. If provided, the affine 
                            matrix will be adjusted to account for the difference
                            between the center of the image and the center of rotation.
        reference_image: The coordinate space that matrix and translation were defined
                         in respect to. If not supplied, the coordinate space of image
                         is used.
        
    Returns:
        A 4x4 MONAI affine matrix.
    """

    assert_itk_regions_match_array(image)
    ndim = image.ndim

    # If there is a reference image, compute an affine matrix that maps the image space to the
    # reference image space.
    if reference_image:
        assert_itk_regions_match_array(reference_image)
        assert image.shape == reference_image.shape, "ITK-MONAI bridge: shape mismatch between image and reference image"
        reference_affine_matrix = compute_reference_space_affine_matrix(image, reference_image)
    else:
        reference_affine_matrix = torch.eye(ndim+1, dtype=torch.float64)

    # Create affine matrix that includes translation
    affine_matrix = torch.eye(ndim+1, dtype=torch.float64)
    affine_matrix[:ndim, :ndim] = torch.tensor(matrix, dtype=torch.float64)
    affine_matrix[:ndim, ndim] = torch.tensor(translation, dtype=torch.float64) 

    # Adjust offset when center of rotation is different from center of the image
    if center_of_rotation:
        offset_matrix, inverse_offset_matrix = compute_offset_matrix(image, center_of_rotation)
        affine_matrix = inverse_offset_matrix @ affine_matrix @ offset_matrix

    # Adjust direction
    direction_matrix, inverse_direction_matrix = compute_direction_matrix(image)
    affine_matrix = inverse_direction_matrix @ affine_matrix @ direction_matrix

    # Adjust based on spacing. It is required because MONAI does not update the 
    # pixel array according to the spacing after a transformation. For example,
    # a rotation of 90deg for an image with different spacing along the two axis
    # will just rotate the image array by 90deg without also scaling accordingly.
    spacing_matrix, inverse_spacing_matrix = compute_spacing_matrix(image)
    affine_matrix = inverse_spacing_matrix @ affine_matrix @ spacing_matrix 

    return affine_matrix @ reference_affine_matrix

def monai_to_itk_affine(image, affine_matrix, center_of_rotation=None):
    """
    Converts a MONAI affine matrix an to ITK affine matrix (2x2 for 2D or 3x3 for
    3D matrix and translation vector). See also 'itk_to_monai_affine'.
    
    Args:
        image: The ITK image object. This is used to extract the spacing and 
               direction information.
        affine_matrix: The 3x3 for 2D or 4x4 for 3D MONAI affine matrix.
        center_of_rotation: The center of rotation. If provided, the affine 
                            matrix will be adjusted to account for the difference
                            between the center of the image and the center of rotation.
        
    Returns:
        The ITK matrix and the translation vector.
    """
    assert_itk_regions_match_array(image)

    # Adjust spacing
    spacing_matrix, inverse_spacing_matrix = compute_spacing_matrix(image)
    affine_matrix = spacing_matrix @ affine_matrix @ inverse_spacing_matrix 

    # Adjust direction
    direction_matrix, inverse_direction_matrix = compute_direction_matrix(image)
    affine_matrix = direction_matrix @ affine_matrix @ inverse_direction_matrix 

    # Adjust offset when center of rotation is different from center of the image
    if center_of_rotation:
        offset_matrix, inverse_offset_matrix = compute_offset_matrix(image, center_of_rotation)
        affine_matrix = offset_matrix @ affine_matrix @ inverse_offset_matrix

    ndim = image.ndim
    matrix = affine_matrix[:ndim, :ndim].numpy()
    translation = affine_matrix[:ndim, ndim].tolist()

    return matrix, translation 


    
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
        if image.ndim == 2: 
            itk_transform.Rotate2D(rotation[0])
        else:
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

    matrix = np.asarray(itk_transform.GetMatrix(), dtype=np.float64)

    return matrix, translation


def itk_affine_resample(image, matrix, translation, center_of_rotation=None, reference_image=None):
    # Translation transform
    itk_transform = itk.AffineTransform[itk.D, image.ndim].New()

    # Set center
    if center_of_rotation:
        itk_transform.SetCenter(center_of_rotation)
    else:
        itk_transform.SetCenter(get_itk_image_center(image))

    # Set matrix and translation
    itk_transform.SetMatrix(itk.matrix_from_array(matrix))
    itk_transform.Translate(translation)

    # Interpolator
    image = image.astype(itk.D)
    interpolator = itk.LinearInterpolateImageFunction.New(image)

    if not reference_image:
        reference_image = image

    # Resample with ITK
    output_image = itk.resample_image_filter(image,
                                             interpolator=interpolator,
                                             transform=itk_transform,
                                             output_parameters_from_image=reference_image) 

    return np.asarray(output_image, dtype=np.float32)


def monai_affine_resample(metatensor, affine_matrix):
    monai_transform = Affine(affine=affine_matrix, padding_mode="zeros", dtype=torch.float64)
    output_tensor, output_affine = monai_transform(metatensor, mode='bilinear')

    return metatensor_to_array(output_tensor)
