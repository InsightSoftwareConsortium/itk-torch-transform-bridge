from itk_torch_ddf_bridge import *
from bridge_utils import remove_border

def test_random_array(ndim):
    print("\nTest: Random array with random spacing, direction and origin, ndim={}".format(ndim))

    # Create image/array with random size and pixel intensities
    s = torch.randint(low=2, high=20, size=(ndim,))
    img = 100 * torch.rand((1, 1, *s.tolist()), dtype=torch.float32)

    # Pad at the edges because ITK and MONAI have different behavior there
    # during resampling
    img = torch.nn.functional.pad(img, pad=ndim*(1, 1))
    ddf = 5 * torch.rand((1, ndim, *img.shape[-ndim:]), dtype=torch.float32) - 2.5

    # Warp with MONAI
    img_resampled = monai_wrap(img, ddf)

    # Create ITK image
    itk_img = itk.GetImageFromArray(img.squeeze().numpy())

    # Set random spacing
    spacing = 3 * np.random.rand(ndim) 
    itk_img.SetSpacing(spacing)

    # Set random direction
    direction = 5 * np.random.rand(ndim, ndim) - 5
    direction = itk.matrix_from_array(direction)
    itk_img.SetDirection(direction)

    # Set random origin
    origin = 100 * np.random.rand(ndim) - 100
    itk_img.SetOrigin(origin)

    # Warp with ITK
    itk_img_resampled = itk_warp(itk_img, ddf.squeeze().numpy())

    # Compare
    print("All close: ", np.allclose(img_resampled, itk_img_resampled, rtol=1e-3, atol=1e-3))
    diff = img_resampled - itk_img_resampled
    print(diff.min(), diff.max())


def test_real_data(filepath):
    print("\nTEST: Real data with random deformation field")
    # Read image
    image = itk.imread(filepath, itk.F)
    image[:] = remove_border(image)
    ndim = image.ndim

    # Random ddf
    ddf = 10 * torch.rand((1, ndim, *image.shape), dtype=torch.float32) - 10

    # Warp with MONAI
    image_tensor = torch.tensor(itk.GetArrayFromImage(image), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    img_resampled = monai_wrap(image_tensor, ddf)

    # Warp with ITK
    itk_img_resampled = itk_warp(image, ddf.squeeze().numpy())

    # Compare
    print("All close: ", np.allclose(img_resampled, itk_img_resampled))
    diff = img_resampled - itk_img_resampled
    print(diff.min(), diff.max())


