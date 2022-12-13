from test_cases import *

test_setting_affine_parameters(filepath = './input/ct_lung_downsampled_reversed.tif')
test_arbitary_center_of_rotation(filepath = './input/ct_lung_downsampled_reversed.tif')
test_registration("./input/fixed_2mm.nii.gz", "./input/moving_2mm.nii.gz")
