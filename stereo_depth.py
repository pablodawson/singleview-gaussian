import sys
sys.path.append("unimatch")
from unimatch.unimatch import UniMatch
import torch
from dataloader.stereo import transforms
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import hflip
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class StereoDepth:
    def __init__(self, device="cuda", checkpoint= "unimatch/checkpoints/things.pth"):
        pass
        self.model = UniMatch(feature_channels=128,
                        num_scales=2,
                        upsample_factor=4,
                        ffn_dim_expansion=4,
                        num_transformer_layers=6,
                        reg_refine=True,
                        task="stereo")

        self.device = device
        checkpoint = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.eval()
        val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]
        self.val_transform = transforms.Compose(val_transform_list)
    
    @torch.no_grad()
    def __call__(self, image1, image2, task='stereo'):
        """Inference on an image pair for optical flow or stereo disparity prediction"""

        padding_factor = 32
        attn_type = 'swin' if task == 'flow' else 'self_swin2d_cross_swin1d'
        attn_splits_list = [2, 8]
        corr_radius_list = [-1, 4]
        prop_radius_list = [-1, 1]
        num_reg_refine = 6 if task == 'flow' else 3


        image1 = np.array(image1).astype(np.float32)
        image2 = np.array(image2).astype(np.float32)

        if len(image1.shape) == 2:  # gray image
            image1 = np.tile(image1[..., None], (1, 1, 3))
            image2 = np.tile(image2[..., None], (1, 1, 3))
        else:
            image1 = image1[..., :3]
            image2 = image2[..., :3]

        sample = {'left': image1, 'right': image2}
        sample = self.val_transform(sample)

        image1 = sample['left'].unsqueeze(0)  # [1, 3, H, W]
        image2 = sample['right'].unsqueeze(0)  # [1, 3, H, W]

        nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                        int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]

        max_inference_size = [384, 768] if task == 'flow' else [640, 960]

        inference_size = [min(max_inference_size[0], nearest_size[0]), min(max_inference_size[1], nearest_size[1])]

        assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
        ori_size = image1.shape[-2:]

        # resize before inference
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                align_corners=True)
            image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                align_corners=True)

        results_dict = self.model(image1, image2,
                            attn_type=attn_type,
                            attn_splits_list=attn_splits_list,
                            corr_radius_list=corr_radius_list,
                            prop_radius_list=prop_radius_list,
                            num_reg_refine=num_reg_refine,
                            task=task,
                            )

        flow_pr = results_dict['flow_preds'][-1]  # [1, 2, H, W] or [1, H, W]

        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            pred_disp = F.interpolate(flow_pr.unsqueeze(1), size=ori_size,
                                    mode='bilinear',
                                    align_corners=True).squeeze(1)  # [1, H, W]
        pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        disp = pred_disp[0].cpu().numpy()

        return disp
        #return Image.fromarray(disp)



if __name__== "__main__":
    stereo_depth = StereoDepth()
    imgL = Image.open("output/images/cam01.png")
    imgR = Image.open("output/images/cam00.png")
    disp = stereo_depth(imgL, imgR)

    disp = (disp - disp.min())/(disp.max() - disp.min()) * 255
    Image.fromarray(disp.astype(np.uint8)).save("disparity.png")