
import sys
sys.path.append("Infusion/depth_inpainting")
from Infusion.depth_inpainting.inference.depth_inpainting_pipeline_half import DepthEstimationInpaintPipeline
import torch
from PIL import Image
import numpy as np
import os

from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)

from transformers import CLIPTextModel, CLIPTokenizer
from geometry_utils import *

def depth_to_saveable( depth):
    return np.clip((depth * (2**12 - 1)),0, 2**16-1).astype(np.uint16)


class SpaceFiller:
    def __init__(self, cameras=None, points=None, colors=None):
        
        dtype = torch.float16
        path = "Johanan0528/Infusion"

        vae = AutoencoderKL.from_pretrained(path,subfolder='vae',torch_dtype=dtype)
        scheduler = DDIMScheduler.from_pretrained(path,subfolder='scheduler',torch_dtype=dtype)
        text_encoder = CLIPTextModel.from_pretrained(path,subfolder='text_encoder',torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained(path,subfolder='tokenizer',torch_dtype=dtype)
        
        unet = UNet2DConditionModel.from_pretrained(path,subfolder="unet",
                                                    in_channels=13, sample_size=96,
                                                    low_cpu_mem_usage=False,
                                                    ignore_mismatched_sizes=True,
                                                    torch_dtype=dtype)
        
        self.pipe = DepthEstimationInpaintPipeline(unet=unet,
                                       vae=vae,
                                       scheduler=scheduler,
                                       text_encoder=text_encoder,
                                       tokenizer=tokenizer,
                                       )

        #self.pipe.enable_vae_slicing()
        self.pipe = self.pipe.to("cuda")
        #self.pipe.enable_xformers_memory_efficient_attention()
        #self.pipe.enable_sequential_cpu_offload()

        self.points = points
        self.colors = colors
        self.cameras = cameras

        #self.fill(cameras)

    def __call__(self, depth, image, mask):
        # Add noise in mask

        noise = np.random.normal(0, 0.1, depth.shape)
        noise = (noise - noise.min())/(noise.max() - noise.min()) * np.percentile(depth, 60)
        mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=2)
        depth[mask == 1] = noise[mask == 1]
        
        inpainted_depth = self.pipe(None,
            denosing_steps=15,
            processing_res = 768,
            match_input_res = False,
            batch_size = 0,
            show_progress_bar = True,
            depth_numpy = depth,
            mask = mask,
            path_to_save = "output",
            colors_png=image,
            blend=True,
            use_mask=True,
            input_image=Image.fromarray(image)
            )
        
        return inpainted_depth


    def fill(self):

        cameras = self.cameras

        init_points = self.points
        init_colors = self.colors
        output_path = "output/inpainted_depths"
        os.makedirs(output_path, exist_ok=True)

        images = cameras.get_imgs()
        masks = cameras.get_masks()
        depths = cameras.get_depths()
        poses = cameras.get_poses()


        for i, pose in enumerate(poses):

            if pose["type"] == "base":
                continue
            
            depth_pil = Image.open(depths[i])
            depth_numpy = np.array(depth_pil) / (2**12 - 1)
            w, h = depth_pil.size
            mask_np = (np.array(Image.open(masks[i]).resize((w,h)))/255).astype(float)

            # Add noise in mask
            noise = np.random.normal(0, 0.1, depth_numpy.shape)
            noise = (noise - noise.min())/(noise.max() - noise.min()) * np.percentile(depth_numpy, 90)
            depth_numpy[mask_np == 1] = noise[mask_np == 1]

            inpainted_depth = self.pipe(images[i],
                denosing_steps=15,
                processing_res = 768,
                match_input_res = False,
                batch_size = 0,
                show_progress_bar = True,
                depth_numpy = depth_numpy,
                mask = mask_np,
                path_to_save = "output",
                colors_png=images[i],
                blend=True,
                use_mask=True
                )
            
            Image.fromarray(depth_to_saveable(inpainted_depth)).save(os.path.join(output_path, os.path.basename(depths[i])))
            
            smaller_mask = cv2.erode(mask_np, np.ones((3,3), np.uint8), iterations=1)
            new_points3d, new_colors = get_reconstruction(np.array(Image.open(images[i])), inpainted_depth, cameras, 
                                                          mask = smaller_mask, max_points = 6000)
            
            self.points = np.concatenate([self.points, new_points3d])
            self.colors = np.concatenate([self.colors, new_colors])

    def get_points_colors(self):
        return self.points, self.colors

