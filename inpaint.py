from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import numpy as np
import cv2


class Inpainter:
    def __init__(self):
        # Image inpainting
        self.image_inpaint_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            custom_pipeline="hyoungwoncho/sd_perturbed_attention_guidance_inpaint",
            torch_dtype=torch.float16,
            safety_checker=None
        )

        device="cuda"
        self.image_inpaint_pipe = self.image_inpaint_pipe.to(device)
        #self.image_inpaint_pipe.enable_vae_slicing()
        self.image_inpaint_pipe.enable_xformers_memory_efficient_attention()
        self.image_inpaint_pipe.enable_sequential_cpu_offload()

    def __call__(self, image, mask, strength=1.0):
        
        if type(image) == np.ndarray:
            image = Image.fromarray(image)
        
        if type(mask) == np.ndarray:
            mask = Image.fromarray(mask)

        output =  self.image_inpaint_pipe(
            "background",
            image=image,
            mask_image=mask,
            num_inference_steps=20,
            guidance_scale=0.0,
            pag_scale=3.0,
            strength = strength,
            pag_applied_layers_index=['u0']
        ).images[0]
        
        return np.array(output)

    def release(self):
        del(self.image_inpaint_pipe)

if __name__ == "__main__":
    inpainter = Inpainter()

    image = Image.open("output/images_aug/aug_1.png")
    depth = Image.open("output/depths_aug/aug_1.png")
    mask = Image.open("output/masks/mask_1.png")

    output = inpainter(image, mask, depth)