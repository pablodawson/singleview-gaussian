# infer.py
# Code by @1ssb
import sys
sys.path.append('DepthAnything/metric_depth')

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from DepthAnything.metric_depth.zoedepth.models.builder import build_model
from DepthAnything.metric_depth.zoedepth.utils.config import get_config
import torch.nn.functional as F


class DepthAnythingMetric:
    def __init__(self):
        config = get_config("zoedepth", "eval", 'nyu')
        config.pretrained_resource = "local::./checkpoints/depth_anything_metric_depth_indoor.pt"
        self.model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()

    def __call__(self, color_image: Image):    
        
        size = color_image.size
        image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

        pred = self.model(image_tensor, dataset="nyu")

        if isinstance(pred, dict):
            pred = pred.get('metric_depth', pred.get('out'))
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        
        depth = F.interpolate(pred, (size[1], size[0]), mode='bilinear', align_corners=False)[0, 0]
        depth = depth.detach().cpu().numpy()
        
        return depth

if __name__ == "__main__":
    estimator = DepthAnythingMetric()
    color_image = Image.open('input/run.jpg')
    resized_pred = estimator(color_image)
    disp = 1 / (resized_pred + 1e-6)
    pred = (disp - disp.min()) / (disp.max() - disp.min())
    Image.fromarray((pred * 255).astype(np.uint8)).save('output/disp.jpg')