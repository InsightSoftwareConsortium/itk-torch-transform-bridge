import itk
import torch
import torch.nn.functional as F

def monai_warp_to_itk_transform(image_fixed: "itk.Image", image_moving:"itk.Image", network_shape: "[int]", network: "torch.nn.Module", **kwargs)->"itk.Transform":
    tensor_fixed, tensor_moving, convert_back = itk_transform_bridge(image_fixed, image_moving, network_shape, phi_type="displacement_field", order="vector_first", **kwargs)
    phi = network(tensor_fixed, tensor_moving)

    return convert_back(phi)

def grid_sample_to_itk_transform(image_fixed: "itk.Image", image_moving:"itk.Image", network_shape: "[int]", network: "torch.nn.Module", align_corners)->"itk.Transform":
    tensor_fixed, tensor_moving, convert_back = itk_transform_bridge(image_fixed, image_moving, network_shape, phi_type="coordinate_field", range=(-1, 1), order="vector_last", align_corners=align_corners)
    phi = network(tensor_fixed, tensor_moving)

    return convert_back(phi)

def itk_transform_bridge(image_fixed: "itk.Image", image_moving:"itk.Image", network_shape: "[int]", phi_type="displacement_field", range=(-1, 1))->"(torch.Tensor, torch.Tensor, Callable[[torch.Tensor], itk.Transform])":
    to_network_space = resampling_transform(image_moving, network_shape)
    from_network_space = resampling_transform(image_fixed, network_shape).GetInverse()
    
    moving_npy = np.array(image_moving)
    fixed_npy = np.array(image_fixed)
    
    # turn images into torch Tensors: add feature and batch dimensions (each of length 1)
    moving_trch = torch.Tensor(moving_npy)[None, None]
    fixed_trch = torch.Tensor(fixed_npy)[None, None]

    
    # Here we resize the input images to the shape expected by the neural network. This affects the 
    # pixel stride as well as the magnitude of the displacement vectors of the resulting displacement field, which
    # convert_back will have to compensate for. 
    
    #TODO: it is crucial to blur before this step if we are downsampling!
    moving_resized = F.interpolate(moving_trch, size=network_shape, mode="trilinear", align_corners=False)
    fixed_resized = F.interpolate(fixed_trch, size=network_shape, mode="trilinear", align_corners=False)
    

    # Create convert_back function

    def convert_back(phi: "torch.Tensor") -> "itk.Transform":
        phi = phi.cpu().detach()
        
        if phi_type == "coordinate_field" and range == (-1, 1):
            # itk.DeformationFieldTransform expects a displacement field, so we subtract off the identity map.
            disp = (phi - )

        dimension = len(network_shape_list)

           
        # We convert the displacement field into an itk Vector Image. 
        scale = torch.Tensor(network_shape_list)

        for _ in network_shape_list:
            scale = scale[:, None]
        disp *= scale

        # disp is a shape [3, H, W, D] tensor with vector components in the order [vi, vj, vk]
        disp_itk_format  = disp.double().numpy()[list(reversed(range(dimension)))].transpose(list(range(1, dimension + 1)) + [0])
        # disp_itk_format is a shape [H, W, D, 3] array with vector components in the order [vk, vj, vi]
        # as expected by itk.

        itk_disp_field = itk.image_from_array(disp_itk_format, is_vector=True)
        
        deformable_transform = itk.DisplacementFieldTransform[(itk.D, dimension)].New()

        deformable_transform.SetDisplacementField(itk_disp_field)
        
        final_transform = itk.CompositeTransform[itk.D, dimension].New()

        final_transform.PrependTransform(from_network_space)
        final_transform.PrependTransform(deformable_transform)
        final_transform.PrependTransform(to_network_space)
        
        return final_transform

    return fixed_resized, moving_resized, convert_back

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
