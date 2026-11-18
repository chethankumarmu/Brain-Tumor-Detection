"""
Grad-CAM: a heatmap showing which parts of the scan the model used most for its decision.
It shows where the model looked. It is not a diagnosis.
"""
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        # Only record when gradients are on; normal predictions run under no_grad and must not be affected.
        if output.requires_grad:
            self.activations = output
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def __call__(self, x, class_idx=None):
        """x: one image tensor (1, 3, H, W). Returns a heatmap (H, W) in [0, 1] and the class used."""
        self.model.eval()
        self.model.zero_grad()
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
            logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def overlay_heatmap(image_rgb_uint8, cam):
    """Blend the heatmap (red = most attention) onto the scan."""
    heat = plt.cm.jet(cam)[..., :3]
    base = image_rgb_uint8.astype(np.float32) / 255.0
    out = 0.55 * base + 0.45 * heat
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def array_to_base64_png(array):
    buf = io.BytesIO()
    plt.imsave(buf, array, format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")
