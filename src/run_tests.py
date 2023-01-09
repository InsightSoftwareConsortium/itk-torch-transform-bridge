from test_cases import *
import test_utils 

test_utils.download_test_data()

filepath1 = str(test_utils.TEST_DATA_DIR / 'copd1_highres_INSP_STD_COPD_img.nii.gz')
filepath2 = str(test_utils.TEST_DATA_DIR / 'copd1_highres_EXP_STD_COPD_img.nii.gz')

test_setting_affine_parameters(filepath=filepath1)
test_arbitary_center_of_rotation(filepath=filepath1)
test_registration(fixed_filepath=filepath1, moving_filepath=filepath2)
