from test_cases import *
import test_utils 

test_utils.download_test_data()

# 2D cases 
filepath0 = str(test_utils.TEST_DATA_DIR / 'CT_2D_head_fixed.mha')
filepath1 = str(test_utils.TEST_DATA_DIR / 'CT_2D_head_moving.mha')

test_setting_affine_parameters(filepath=filepath0)
test_arbitary_center_of_rotation(filepath=filepath0)
test_registration(fixed_filepath=filepath0, moving_filepath=filepath1)

# 3D cases
filepath2 = str(test_utils.TEST_DATA_DIR / 'copd1_highres_INSP_STD_COPD_img.nii.gz')
filepath3 = str(test_utils.TEST_DATA_DIR / 'copd1_highres_EXP_STD_COPD_img.nii.gz')

test_setting_affine_parameters(filepath=filepath2)
test_arbitary_center_of_rotation(filepath=filepath2)
test_registration(fixed_filepath=filepath2, moving_filepath=filepath3)
