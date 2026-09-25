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

        Parameters
        ----------
        image_tensor:
            Preprocessed image tensor of shape [1, 3, 224, 224].

        class_index:
            Disease class index whose activation should be explained.

        Returns
        -------
        cam:
            Normalized Grad-CAM heatmap as a NumPy array.

        output:
            Raw classifier output tensor.
        """

        self.model.zero_grad()

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

        return (
            cam.detach().cpu().numpy(),
            output.detach(),
        )

    def close(self):
        """
        Remove registered PyTorch hooks.
        """

        self.forward_handle.remove()
        self.backward_handle.remove()