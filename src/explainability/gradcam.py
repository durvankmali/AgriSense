import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM implementation for AgriSense.

    Uses gradients flowing into the selected target layer
    to identify image regions that influenced a prediction.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self.save_activations
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self.save_gradients
        )

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(
    self,
    image_tensor: torch.Tensor,
    class_index: int,
    ):
        """
        Generate a normalized Grad-CAM heatmap.
        """

        self.model.zero_grad(set_to_none=True)

        output = self.model(image_tensor)

        target_score = output[:, class_index].sum()

        target_score.backward()

        gradients = self.gradients
        activations = self.activations

        # Global average pooling of gradients
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # Weighted combination of activation maps
        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        # Keep only positive influence
        cam = torch.relu(cam)

        # Resize CAM to input image dimensions
        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # Remove batch/channel dimensions
        cam = cam.squeeze()

        # Normalize to 0–1
        cam_min = cam.min()
        cam_max = cam.max()

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min + 1e-8
        )

        # Move only the final CAM to CPU.
        cam_numpy = cam.detach().cpu().numpy()

        # The raw model output is not needed by the production pipeline.
        del target_score
        del weights
        del cam
        del gradients
        del activations
        del output

        # Clear hook references so they don't retain tensors.
        self.gradients = None
        self.activations = None

        return cam_numpy

    def close(self):
        """
        Remove registered PyTorch hooks.
        """

        self.forward_handle.remove()
        self.backward_handle.remove()