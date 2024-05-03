import torch
from Marigold.marigold import MarigoldPipeline
import numpy as np

def depth_to_saveable( depth):
    return np.clip((depth * (2**12 - 1)),0, 2**16-1).astype(np.uint16)

class MarigoldDepth:
    def __init__(self, device="cuda", seed=0):
        dtype = torch.float16
        variant = "fp16"

        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

        self.pipe = MarigoldPipeline.from_pretrained(
        "prs-eth/marigold-lcm-v1-0", variant=variant, torch_dtype=dtype
        ).to(device)
    
    def __call__(self, input_image):
        pipe_out = self.pipe(
                input_image,
                denoising_steps=4,
                ensemble_size=2,
                processing_res=768,
                match_input_res=True,
                show_progress_bar=True,
                generator=self.generator,
            )
        
        return pipe_out.depth_np
        
if __name__ == "__main__":
    from PIL import Image

    depth_predictor = MarigoldDepth()
    image = Image.open("input/0001.png")
    depth = depth_predictor(image)
    depth_save = depth_to_saveable(depth)
    Image.fromarray(depth_save).save("output/depth.png")

