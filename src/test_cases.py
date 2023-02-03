import itk
import numpy as np 
from itk_torch_affine_matrix_bridge import *

import copy

def test_setting_affine_parameters(filepath):
    print("\nTEST: Setting affine parameters, center of rotation is center of the image")
    # Read image
    image = itk.imread(filepath, itk.F)
    image[:] = remove_border(image)
    ndim = image.ndim

    # Affine parameters
    translation = [65.2, -50.2, 33.9][:ndim]
    rotation = [0.78539816339, 1.0, -0.66][:ndim]
    scale = [2.0, 1.5, 3.2][:ndim]
    shear = [0, 1, 1.6] # axis1, axis2, coeff

    # Spacing 
    spacing = np.array([1.2, 1.5, 2.0])[:ndim]
    image.SetSpacing(spacing)

    # ITK
    matrix, translation = create_itk_affine_from_parameters(image, translation=translation, rotation=rotation, scale=scale, shear=shear)
    output_array_itk = itk_affine_resample(image, matrix=matrix, translation=translation)

    # MONAI
    metatensor = image_to_metatensor(image)
    affine_matrix_for_monai = itk_to_monai_affine(image, matrix=matrix, translation=translation)
    output_array_monai = monai_affine_resample(metatensor, affine_matrix=affine_matrix_for_monai)

    ###########################################################################
    # Make sure that the array conversion of the inputs is the same 
    input_array_monai = metatensor_to_array(metatensor)
    assert(np.array_equal(input_array_monai, np.asarray(image)))

    # Compare outputs
    print("MONAI-ITK: ", np.allclose(output_array_monai, output_array_itk))

    diff_output = output_array_monai - output_array_itk
    print("[Min, Max] MONAI: [{}, {}]".format(output_array_monai.min(), output_array_monai.max()))
    print("[Min, Max] ITK: [{}, {}]".format(output_array_itk.min(), output_array_itk.max()))
    print("[Min, Max] diff: [{}, {}]".format(diff_output.min(), diff_output.max()))

    # Write 
    # itk.imwrite(itk.GetImageFromArray(diff_output), "./output/diff.tif")
    # itk.imwrite(itk.GetImageFromArray(output_array_monai), "./output/output_monai.tif")
    # itk.imwrite(itk.GetImageFromArray(output_array_itk), "./output/output_itk.tif")
    ###########################################################################



def test_arbitary_center_of_rotation(filepath):
    print("\nTEST: affine matrix with arbitary center of rotation")
    # Read image
    image = itk.imread(filepath, itk.F)
    image[:] = remove_border(image)
    ndim = image.ndim

    # ITK matrix (3x3 affine matrix)
    matrix = np.array([[0.55915995, 0.50344867, 0.43208387],
                       [0.01133669, 0.82088571, 0.86841365],
                       [0.30478496, 0.94998986, 0.32742505]])[:ndim, :ndim]
    translation = [54.0, 2.7, -11.9][:ndim]

    # Spatial properties
    center_of_rotation = [-32.3, 125.1, 0.7][:ndim]
    origin = [1.6, 0.5, 2.0][:ndim]
    spacing = np.array([1.2, 1.5, 0.6])[:ndim]

    image.SetSpacing(spacing)
    image.SetOrigin(origin)

    # ITK
    output_array_itk = itk_affine_resample(image, matrix=matrix, translation=translation, center_of_rotation=center_of_rotation)

    # MONAI
    metatensor = image_to_metatensor(image)
    affine_matrix_for_monai = itk_to_monai_affine(image, matrix=matrix, translation=translation, center_of_rotation=center_of_rotation)
    output_array_monai = monai_affine_resample(metatensor, affine_matrix=affine_matrix_for_monai)

    # Make sure that the array conversion of the inputs is the same 
    input_array_monai = metatensor_to_array(metatensor)
    assert(np.array_equal(input_array_monai, np.asarray(image)))
    
    ###########################################################################
    # Compare outputs
    print("MONAI-ITK: ", np.allclose(output_array_monai, output_array_itk))

    diff_output = output_array_monai - output_array_itk
    print("[Min, Max] MONAI: [{}, {}]".format(output_array_monai.min(), output_array_monai.max()))
    print("[Min, Max] ITK: [{}, {}]".format(output_array_itk.min(), output_array_itk.max()))
    print("[Min, Max] diff: [{}, {}]".format(diff_output.min(), diff_output.max()))
    ###########################################################################


def test_monai_to_itk(filepath):
    print("\nTEST: MONAI affine matrix -> ITK matrix + translation vector -> transform")
    # Read image
    image = itk.imread(filepath, itk.F)

    image[:] = remove_border(image)
    ndim = image.ndim

    # MONAI affine matrix 
    affine_matrix = torch.eye(ndim+1, dtype=torch.float64)
    affine_matrix[:ndim, :ndim] = torch.tensor([[0.55915995, 0.50344867, 0.43208387],
                                                [0.01133669, 0.82088571, 0.86841365],
                                                [0.30478496, 0.94998986, 0.32742505]],
                                                dtype=torch.float64)[:ndim, :ndim]

    affine_matrix[:ndim, ndim] = torch.tensor([54.0, 2.7, -11.9],
                                              dtype=torch.float64)[:ndim]

    # Spatial properties
    center_of_rotation = [-32.3, 125.1, 0.7][:ndim]
    origin = [1.6, 0.5, 2.0][:ndim]
    spacing = np.array([1.2, 1.5, 0.6])[:ndim]

    image.SetSpacing(spacing)
    image.SetOrigin(origin)


    # ITK
    matrix, translation = monai_to_itk_affine(image, affine_matrix=affine_matrix, center_of_rotation=center_of_rotation)
    output_array_itk = itk_affine_resample(image, matrix=matrix, translation=translation, center_of_rotation=center_of_rotation)

    # MONAI
    metatensor = image_to_metatensor(image)
    output_array_monai = monai_affine_resample(metatensor, affine_matrix=affine_matrix)

    # Make sure that the array conversion of the inputs is the same 
    input_array_monai = metatensor_to_array(metatensor)
    assert(np.array_equal(input_array_monai, np.asarray(image)))
    
    ###########################################################################
    # Compare outputs
    print("MONAI-ITK: ", np.allclose(output_array_monai, output_array_itk))

    diff_output = output_array_monai - output_array_itk
    print("[Min, Max] MONAI: [{}, {}]".format(output_array_monai.min(), output_array_monai.max()))
    print("[Min, Max] ITK: [{}, {}]".format(output_array_itk.min(), output_array_itk.max()))
    print("[Min, Max] diff: [{}, {}]".format(diff_output.min(), diff_output.max()))
    ###########################################################################


def test_cyclic_conversion(filepath):
    print("\nTEST: matrix + translation -> affine_matrix -> matrix + translation")
    image = itk.imread(filepath, itk.F)
    image[:] = remove_border(image)
    ndim = image.ndim

    # ITK matrix (3x3 affine matrix)
    matrix = np.array([[2.90971094, 1.18297296, 2.60008784],
                       [0.29416137, 0.10294283, 2.82302616],
                       [1.70578374, 1.39706003, 2.54652029]])[:ndim, :ndim]

    translation = [-29.05463245,  35.27116398,  48.58759597][:ndim]

    # Spatial properties
    center_of_rotation = [-27.84789587, -60.7871084 , 42.73501932][:ndim]
    origin = [8.10416794, 5.4831944, 0.49211025][:ndim]
    spacing = np.array([0.7, 3.2, 1.3])[:ndim]

    image.SetSpacing(spacing)
    image.SetOrigin(origin)

    affine_matrix = itk_to_monai_affine(image, matrix=matrix, translation=translation, center_of_rotation=center_of_rotation)

    matrix_result, translation_result = monai_to_itk_affine(image, affine_matrix=affine_matrix, center_of_rotation=center_of_rotation) 

    print("Matrix cyclic conversion: ", np.allclose(matrix, matrix_result))
    print("Translation cyclic conversion: ", np.allclose(translation, translation_result))

def test_use_reference_space(ref_filepath, filepath):
    print("\nTEST: calculate affine matrix for an image based on a reference space")
    # Read the images
    image = itk.imread(filepath, itk.F)
    image[:] = remove_border(image)
    ndim = image.ndim

    ref_image = itk.imread(ref_filepath, itk.F)

    # Set arbitary origin, spacing, direction for both of the images
    image.SetSpacing([1.2, 2.0, 1.7][:ndim])
    ref_image.SetSpacing([1.9, 1.5, 1.3][:ndim])
    
    direction = np.eye(3, dtype=np.float64)
    direction[0, 0] = 0.68
    direction[1, 1] = 1.05
    direction[2, 2] = 1.83
    image.SetDirection(direction[:ndim, :ndim])

    ref_direction = np.eye(3, dtype=np.float64)
    ref_direction[0, 0] = 1.25
    ref_direction[1, 1] = 0.99 
    ref_direction[2, 2] = 1.50
    ref_image.SetDirection(ref_direction[:ndim, :ndim])

    image.SetOrigin([57.3, 102.0, -20.9][:ndim])
    ref_image.SetOrigin([23.3, -0.5, 23.7][:ndim])

    # Set affine parameters
    matrix = np.array([[0.55915995, 0.50344867, 0.43208387],
                       [0.01133669, 0.82088571, 0.86841365],
                       [0.30478496, 0.94998986, 0.32742505]])[:ndim, :ndim]
    translation = [54.0, 2.7, -11.9][:ndim]
    center_of_rotation = [-32.3, 125.1, 0.7][:ndim]

    # Resample using ITK 
    output_array_itk = itk_affine_resample(image, matrix=matrix, translation=translation, center_of_rotation=center_of_rotation, reference_image=ref_image)

    # MONAI
    metatensor = image_to_metatensor(image)
    affine_matrix_for_monai = itk_to_monai_affine(image, matrix=matrix, translation=translation, center_of_rotation=center_of_rotation, reference_image=ref_image)
    output_array_monai = monai_affine_resample(metatensor, affine_matrix=affine_matrix_for_monai)

    # Compare outputs
    print("MONAI equals ITK: ", np.allclose(output_array_monai, output_array_itk))

    diff_output = output_array_monai - output_array_itk
    print("[Min, Max] MONAI: [{}, {}]".format(output_array_monai.min(), output_array_monai.max()))
    print("[Min, Max] ITK: [{}, {}]".format(output_array_itk.min(), output_array_itk.max()))
    print("[Min, Max] diff: [{}, {}]".format(diff_output.min(), diff_output.max()))

    itk.imwrite(itk.GetImageFromArray(diff_output), "./output/diff.tif")
    itk.imwrite(itk.GetImageFromArray(output_array_monai), "./output/output_monai.tif")
    itk.imwrite(itk.GetImageFromArray(output_array_itk), "./output/output_itk.tif")

    
