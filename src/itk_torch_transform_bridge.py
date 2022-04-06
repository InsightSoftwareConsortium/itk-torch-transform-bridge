import itk
import torch

def itk_transform_bridge(image_fixed: "itk.Image", image_moving:"itk.Image", network_shape: "[int]", **kwargs)->"(torch.Tensor, torch.Tensor, Callable[torch.Tensor, itk.Transform])":
    # Convert images to tensors

    # ...

    # Create convert_back function

    def convert_back(phi: "torch.Tensor") -> "itk.Transform"):
        
        return itk.CompositeTransform(some_stuff)

    return tensor_fixed, tensor_moving, convert_back
