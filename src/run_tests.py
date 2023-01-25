from test_cases import *
import test_utils

test_utils.download_test_data()

filepath_2D = str(test_utils.TEST_DATA_DIR / 'CT_2D_head_fixed.mha')
filepath_3D = str(test_utils.TEST_DATA_DIR / 'copd1_highres_INSP_STD_COPD_img.nii.gz')

# 2D cases
test_random_array(ndim=2)
test_real_data(filepath=filepath_2D)

# 3D cases
test_random_array(ndim=3)
test_real_data(filepath=filepath_3D)


