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


