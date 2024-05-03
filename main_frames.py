import os
from extract import extract_from_video
from argparse import ArgumentParser
from geometry import *
from geometry_utils import *
from create_dataset import create_colmap_dataset
from tqdm import tqdm
from augment import DataAugmentor
import shutil
from flow_estimation import FlowEstimator
from PIL import Image
from space_filler import SpaceFiller

def depth_to_saveable( depth):
    return np.clip((depth * (2**10 - 1)),0, 2**16-1).astype(np.uint16)

def main():
    parser = ArgumentParser()
    parser.add_argument('--video_path', '-s', type=str, help='Path to the video')
    parser.add_argument('--output_path', '-o', type=str, default="output", help='Path to the output folder')
    parser.add_argument('--start_time', type=int, default=0, help='Start time in seconds')
    parser.add_argument('--duration', type=int, default=2, help='Duration in seconds')
    parser.add_argument('--baseline', type=float, default=0.065, help='Baseline distance between the two cameras')
    parser.add_argument('--augment', type=bool, default=True, help='Add data augmentation')
    parser.add_argument('--fov', type=float, default=85, help='Field of view of the cameras')
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    
    
    args = parser.parse_args()
    args.weights =  "gmflow/pretrained/gmflow_things-e9887eda.pth"

    args.video_path = "gabbie.mp4"
    os.makedirs(args.output_path, exist_ok=True)
    
    cameras = Cameras(fov = args.fov, baseline=args.baseline)
    cameras.create_poses(theta = np.pi/26, offset=args.baseline/2, augment=args.augment)
    
    extract_from_video(args.video_path, args.start_time, args.duration, cameras)    

    flow_estimator = FlowEstimator(args.weights, args.device, fov=args.fov)
    
    frame = "0001.png"

    ref_path = os.path.join(args.output_path, "images")
 
    
    os.makedirs(ref_path, exist_ok=True)

    all_cams = [f for f in os.listdir(args.output_path) if os.path.isdir(os.path.join(args.output_path, f)) and f.startswith("cam")]

    for cam in sorted(all_cams):
        shutil.copy(os.path.join(args.output_path, cam, frame), os.path.join(ref_path, cam + ".png"))
        cameras.add_img(os.path.join(ref_path, cam + ".png"))
        cameras.add_mask("dummy")
    
    depthL, depthR = flow_estimator.get_depth(np.array(Image.open(os.path.join(ref_path, "cam00.png"))), 
                                            np.array(Image.open(os.path.join(ref_path, "cam01.png"))))    

    depths_path = os.path.join(args.output_path, "depths")
    os.makedirs(depths_path, exist_ok=True)

    depth_path = os.path.join(depths_path, "depthL.png")
    Image.fromarray(depth_to_saveable(depthL)).save(depth_path)
    cameras.add_depth(depth_path)
    

    depth_path = os.path.join(depths_path, "depthR.png")
    Image.fromarray(depth_to_saveable(depthR)).save(depth_path)
    cameras.add_depth(depth_path)

    colorL = np.array(Image.open(os.path.join(ref_path, "cam00.png")))

    points3d, colors = get_reconstruction(colorL, depthL, cameras)

    if args.augment:
        augmentor = DataAugmentor(fov = args.fov, inpaint=True)

        mask_path = os.path.join(args.output_path, "masks")
        os.makedirs(mask_path, exist_ok=True)
        
        augmentor.augment_images(cameras, mask_save_dir=mask_path)

        space_filler = SpaceFiller(cameras, points=points3d, colors=colors)
        new_points, new_colors = space_filler.get_points_colors()

    recon_path = os.path.join(args.output_path, "frames")
    create_colmap_dataset(recon_path, new_points, new_colors, cameras, offset=0, frame=frame)

if __name__ == '__main__':
    main()