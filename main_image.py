import os
from argparse import ArgumentParser
from geometry import *
from geometry_utils import *
from create_dataset import create_colmap_dataset
from augment import DataAugmentor
from PIL import Image
from depth_predictor import DepthAnythingMetric
#from marigold_depth import MarigoldDepth
import torch


def depth_to_saveable( depth):
    return np.clip((depth * (2**13 - 1)),0, 2**16-1).astype(np.uint16)

def main():
    parser = ArgumentParser()
    parser.add_argument('--image_path', '-s', type=str, help='Path to the video')
    parser.add_argument('--output_path', '-o', type=str, default="output", help='Path to the output folder')
    parser.add_argument('--fov', type=float, default=85, help='Field of view of the cameras')
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    args = parser.parse_args()

    #estimator = MarigoldDepth()
    estimator = DepthAnythingMetric()

    args.image_path = "input/scale.jpg"
    os.makedirs(args.output_path, exist_ok=True)
    
    # Create cameras
    cameras = Cameras(fov = args.fov)
    
    cameras.create_poses(theta = np.pi/15, offset=0.7) # TODO: Use look-at poses instead
    
    # Save init depth/image
    image = Image.open(args.image_path)
    image.save(os.path.join(args.output_path, "init_image.png"))
    
    #max_dist = 100.0
    depth_pred = estimator(image) #* max_dist
    depth_save = depth_to_saveable(depth_pred)

    depth_path = os.path.join(args.output_path, "init_depth.png")
    Image.fromarray(depth_save).save(depth_path)

    del(estimator)
    torch.cuda.empty_cache()
    
    # Paths for augmented images/depths
    ref_path = os.path.join(args.output_path, "images")
    os.makedirs(ref_path, exist_ok=True)

    depths_path = os.path.join(args.output_path, "depths")
    os.makedirs(depths_path, exist_ok=True)
    
    augmentor = DataAugmentor(fov = args.fov)
    
    augmentor.augment_images(cameras, output_path = args.output_path)
    points, colors = augmentor.get_points()

    create_colmap_dataset(args.output_path, points, colors, cameras, offset=0)

if __name__ == '__main__':
    main()