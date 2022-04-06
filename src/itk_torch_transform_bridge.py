import itk
import torch

def monai_warp_to_itk_transform(image_fixed: "itk.Image", image_moving:"itk.Image", network_shape: "[int]", network: "torch.nn.Module", **kwargs)->"itk.Transform":
    tensor_fixed, tensor_moving, convert_back = itk_transform_bridge(image_fixed, image_moving, network_shape, phi_type="displacement_field", order="vector_first", **kwargs)
    phi = network(tensor_fixed, tensor_moving)

    return convert_back(phi)

def grid_sample_to_itk_transform(image_fixed: "itk.Image", image_moving:"itk.Image", network_shape: "[int]", network: "torch.nn.Module", align_corners)->"itk.Transform":
    tensor_fixed, tensor_moving, convert_back = itk_transform_bridge(image_fixed, image_moving, network_shape, phi_type="coordinate_field", range=(-1, 1), order="vector_last", align_corners=align_corners)
    phi = network(tensor_fixed, tensor_moving)

    return convert_back(phi)

def itk_transform_bridge(image_fixed: "itk.Image", image_moving:"itk.Image", network_shape: "[int]", **kwargs)->"(torch.Tensor, torch.Tensor, Callable[[torch.Tensor], itk.Transform])":
    # Convert images to tensors

    # ...

    # Create convert_back function

    def convert_back(phi: "torch.Tensor") -> "itk.Transform":
        
        return itk.CompositeTransform(some_stuff)

    return tensor_fixed, tensor_moving, convert_back

def resampling_transform(image, shape) -> itk.Transform:
    
    imageType = itk.template(image)[0][itk.template(image)[1]]
    
    dummy_image = itk.image_from_array(np.zeros(tuple(reversed(shape)), dtype=itk.array_from_image(image).dtype))
    if len(shape) == 2:
        transformType = itk.MatrixOffsetTransformBase[itk.D, 2, 2]
    else:
        transformType = itk.VersorRigid3DTransform[itk.D]
    initType = itk.CenteredTransformInitializer[transformType, imageType, imageType]
    initializer = initType.New()
    initializer.SetFixedImage(dummy_image)
    initializer.SetMovingImage(image)
    transform = transformType.New()
    
    initializer.SetTransform(transform)
    initializer.InitializeTransform()
    
    if len(shape) == 3:
        transformType = itk.CenteredAffineTransform[itk.D, 3]
        t2 = transformType.New()
        t2.SetCenter(transform.GetCenter())
        t2.SetOffset(transform.GetOffset())
        transform = t2
    m = transform.GetMatrix()
    m_a = itk.array_from_matrix(m)
    
    input_shape = image.GetLargestPossibleRegion().GetSize()
    
    for i in range(len(shape)):
    
        m_a[i, i] = image.GetSpacing()[i] * (input_shape[i] / shape[i])
    
    m_a = itk.array_from_matrix(image.GetDirection()) @ m_a 
    
    transform.SetMatrix(itk.matrix_from_array(m_a))
    
    return transform
