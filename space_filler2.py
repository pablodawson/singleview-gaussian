import sys
sys.path.append("invisible-stitch")
from utils.models import get_zoe_dc_model, infer_with_pad
from huggingface_hub import hf_hub_download
import torch
from PIL import Image
import numpy as np

class SpaceFiller2:
    def __init__(self, device="cuda"):
        self.model = get_zoe_dc_model(ckpt_path=hf_hub_download(repo_id="paulengstler/invisible-stitch", filename="invisible-stitch.pt")).to(device)
        self.device = device
    
    def __call__(self, image, depth, mask, scaling=1.0):
        image = image.permute(2, 0, 1).float() / 255.0
        x = torch.cat([image[None, ...], depth[None, None, ...] / (float(scaling) * 10.0), mask[None, None]], dim=1).to(self.device)
        
        out = infer_with_pad(self.model, x)
        out_flip = infer_with_pad(self.model, torch.flip(x, dims=[3]))
        out = (out + torch.flip(out_flip, dims=[3])) / 2

        pred_depth = float(scaling) * out

        return torch.nn.functional.interpolate(pred_depth, image.shape[-2:], mode='bilinear', align_corners=True)[0, 0]

if __name__=="__main__":
    image = torch.tensor(np.array(Image.open("output/init_image.png").resize((256,256))))
    depth = torch.tensor(np.array(Image.open("output/init_depth.png").resize((256,256)))).float() / (2**13 - 1)

    space_filler = SpaceFiller2()
    pred_depth = space_filler(image, depth, torch.ones_like(depth))
    pass