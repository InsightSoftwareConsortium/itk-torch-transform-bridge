import itk
import numpy as np 
from itk_torch_affine_matrix_bridge import *


def test_setting_affine_parameters(filepath):
    print("\nTEST: Setting affine parameters, center of rotation is center of the image")
    # Affine parameters
    translation = [65.2, -50.2, 33.9]
    rotation = [0.78539816339, 1.0, -0.66]
    scale = [2.0, 1.5, 3.2]
    shear = [0, 2, 1.6] # axis1, axis2, coeff

    # Read image
    spacing = np.array([1.2, 1.5, 2.0])
    image = itk.imread(filepath, itk.F)
    image[:, :, :] = remove_border(image)
    image.SetSpacing(spacing)

    # ITK
    matrix, translation = create_itk_affine_from_parameters(image, translation=translation, rotation=rotation, scale=scale, shear=shear)
    output_array_itk = transform_affinely_with_itk(image, matrix=matrix, translation=translation)

    # MONAI
    metatensor = image_to_metatensor(image)
    affine_matrix_for_monai = itk_to_monai_affine(image, matrix=matrix, translation=translation)
    output_array_monai = transform_affinely_with_monai(metatensor, affine_matrix=affine_matrix_for_monai)

    # Transformix
    output_array_transformix = transform_affinely_with_transformix(image, matrix=matrix, translation=translation)

    ###########################################################################
    # Make sure that the array conversion of the inputs is the same 
    input_array_monai = metatensor_to_array(metatensor)
    assert(np.array_equal(input_array_monai, np.asarray(image)))

    # Compare outputs
    print("MONAI-ITK: ", np.allclose(output_array_monai, output_array_itk))
    print("ITK-Transformix: ", np.allclose(output_array_itk, output_array_transformix))

    diff_output = output_array_monai - output_array_itk
    print("[Min, Max] MONAI: [{}, {}]".format(output_array_monai.min(), output_array_monai.max()))
    print("[Min, Max] ITK: [{}, {}]".format(output_array_itk.min(), output_array_itk.max()))
    print("[Min, Max] diff: [{}, {}]".format(diff_output.min(), diff_output.max()))

    # Write 
    # itk.imwrite(itk.GetImageFromArray(diff_output), "./output/diff.tif")
    # itk.imwrite(itk.GetImageFromArray(output_array_monai), "./output/output_monai.tif")
    # itk.imwrite(itk.GetImageFromArray(output_array_itk), "./output/output_itk.tif")
    # itk.imwrite(itk.GetImageFromArray(output_array_transformix), "./output/output_transformix.tif")
    ###########################################################################



def test_arbitary_center_of_rotation(filepath):
    print("\nTEST: affine matrix with arbitary center of rotation")
    # ITK matrix (3x3 affine matrix)
    matrix = np.array([[0.55915995, 0.50344867, 0.43208387],
                       [0.01133669, 0.82088571, 0.86841365],
                       [0.30478496, 0.94998986, 0.32742505]])
    matrix = itk.matrix_from_array(matrix)
    translation = [54.0, 2.7, -11.9]

    # Spatial properties
    center_of_rotation = [-32.3, 125.1, 0.7]
    origin = [1.6, 0.5, 2.0]
    spacing = np.array([1.2, 1.5, 0.6])

    # Read image
    image = itk.imread(filepath, itk.F)
    image[:, :, :] = remove_border(image)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)

    # ITK
    output_array_itk = transform_affinely_with_itk(image, matrix=matrix, translation=translation, center_of_rotation=center_of_rotation)

    # MONAI
    metatensor = image_to_metatensor(image)
    affine_matrix_for_monai = itk_to_monai_affine(image, matrix=matrix, translation=translation, center_of_rotation=center_of_rotation)
    output_array_monai = transform_affinely_with_monai(metatensor, affine_matrix=affine_matrix_for_monai)

    # Transformix
    output_array_transformix = transform_affinely_with_transformix(image, matrix=matrix, translation=translation, center_of_rotation=center_of_rotation)

    # Make sure that the array conversion of the inputs is the same 
    input_array_monai = metatensor_to_array(metatensor)
    assert(np.array_equal(input_array_monai, np.asarray(image)))
    
    ###########################################################################
    # Compare outputs
    print("MONAI-ITK: ", np.allclose(output_array_monai, output_array_itk))
    print("ITK-Transformix: ", np.allclose(output_array_itk, output_array_transformix))

    diff_output = output_array_monai - output_array_itk
    print("[Min, Max] MONAI: [{}, {}]".format(output_array_monai.min(), output_array_monai.max()))
    print("[Min, Max] ITK: [{}, {}]".format(output_array_itk.min(), output_array_itk.max()))
    print("[Min, Max] diff: [{}, {}]".format(diff_output.min(), diff_output.max()))
    ###########################################################################


def test_registration(fixed_filepath, moving_filepath):
    print("\nTEST: registration with direction different than identity")
    # Read images
    fixed_image = itk.imread(fixed_filepath, itk.F)
    moving_image = itk.imread(moving_filepath, itk.F)

    # MONAI seems to have different interpolation behavior at the borders, and
    # no option matches exactly ITK/Elastix. Hence, we pad to allow for direct
    # numerical comparison later at the outputs.
    fixed_image[:, :, :] = remove_border(fixed_image)
    moving_image[:, :, :] = remove_border(moving_image)

    # Default Affine Parameter Map
    parameter_object = itk.ParameterObject.New()
    affine_parameter_map = parameter_object.GetDefaultParameterMap('affine', 4)
    affine_parameter_map['ResampleInterpolator'] = ['FinalLinearInterpolator']
    parameter_object.AddParameterMap(affine_parameter_map)

    # Register 
    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image, parameter_object=parameter_object)

    # Extract useful transformation parameters
    parameter_map = result_transform_parameters.GetParameterMap(0)

    center_of_rotation = np.array(parameter_map['CenterOfRotationPoint'], dtype=float).tolist()

    transform_parameters = np.array(parameter_map['TransformParameters'], dtype=float)
    matrix = transform_parameters[:9].reshape(3, 3)
    matrix = itk.matrix_from_array(matrix)
    translation = transform_parameters[-3:].tolist()
    
    # Resample using ITK 
    output_array_itk = transform_affinely_with_itk(moving_image, matrix=matrix, translation=translation, center_of_rotation=center_of_rotation)

    # MONAI
    metatensor = image_to_metatensor(moving_image)
    affine_matrix_for_monai = itk_to_monai_affine(moving_image, matrix=matrix, translation=translation, center_of_rotation=center_of_rotation)
    output_array_monai = transform_affinely_with_monai(metatensor, affine_matrix=affine_matrix_for_monai)


    ###########################################################################
    # Compare outputs
    print("ITK equals result: ", np.allclose(output_array_itk, np.asarray(result_image)))
    print("MONAI equals result: ", np.allclose(output_array_monai, np.asarray(result_image)))
    print("MONAI equals ITK: ", np.allclose(output_array_monai, output_array_itk))

    # diff = output_array_monai - np.asarray(result_image)
    # itk.imwrite(itk.GetImageFromArray(diff), "./output/diff.tif")
    # itk.imwrite(itk.GetImageFromArray(output_array_monai), "./output/monai.tif")
    # itk.imwrite(itk.GetImageFromArray(output_array_itk), "./output/itk.tif")
    # itk.imwrite(itk.GetImageFromArray(itk.GetArrayFromImage(result_image)), "./output/result.tif")

    # transformed_image_itk = itk.GetImageFromArray(output_array_itk)
    # transformed_image_itk.SetSpacing(result_image.GetSpacing())
    # transformed_image_itk.SetOrigin(result_image.GetOrigin())


    # itk.imwrite(fixed_image, "./output/fixed_updated_spacing.nii.gz")
    # itk.imwrite(moving_image, "./output/moving_updated_spacing.nii.gz")
    # itk.imwrite(result_image, "./output/registered.nii.gz")
    # itk.imwrite(transformed_image_itk, "./output/transformed_itk.nii.gz")
    ###########################################################################
