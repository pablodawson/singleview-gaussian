import cv2
import os
import sys
sys.path.append("dust3r")

from extract import extract_from_video
from argparse import ArgumentParser
from geometry import *
from geometry_utils import *
from create_dataset import convertdynerftocolmapdb
from tqdm import tqdm
from augment import DataAugmentor
import shutil
from flow_estimation import FlowEstimator
from PIL import Image

def depth_to_saveable( depth):
    return np.clip((depth * (2**12 - 1)),0, 2**16-1).astype(np.uint16)

def main():
    parser = ArgumentParser()
    parser.add_argument('--video_path', '-s', type=str, help='Path to the video')
    parser.add_argument('--output_path', '-o', type=str, default="output", help='Path to the output folder')
    parser.add_argument('--start_time', type=int, default=0, help='Start time in seconds')
    parser.add_argument('--duration', type=int, default=2, help='Duration in seconds')
    parser.add_argument('--baseline', type=float, default=0.065, help='Baseline distance between the two cameras')
    parser.add_argument('--augment', type=bool, default=True, help='Add data augmentation')
    parser.add_argument('--fov', type=float, default=85, help='Field of view of the cameras')
    # Dust3r
    #parser.add_argument("--weights", type=str, required=True, help="path to the model weights")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    
    
    args = parser.parse_args()
    args.weights =  "gmflow/pretrained/gmflow_things-e9887eda.pth"

    args.video_path = "sav.mp4"
    os.makedirs(args.output_path, exist_ok=True)
    
    cameras = Cameras(fov = args.fov, baseline=args.baseline)
    cameras.create_poses(theta = np.pi/16, offset=args.baseline/2)
    
    extract_from_video(args.video_path, args.start_time, args.duration, cameras)    

    flow_estimator = FlowEstimator(args.weights, args.device, fov=args.fov)

    frame = "0001.png"
    
    ref_path = os.path.join(args.output_path, "images")
    os.makedirs(ref_path, exist_ok=True)

    all_cams = [f for f in os.listdir(args.output_path) if os.path.isdir(os.path.join(args.output_path, f)) and f.startswith("cam")]

    for cam in sorted(all_cams):
        shutil.copy(os.path.join(args.output_path, cam, frame), os.path.join(ref_path, cam + ".png"))
        cameras.add_img(os.path.join(ref_path, cam + ".png"))
    
    depthL, depthR = flow_estimator.get_depth(np.array(Image.open(os.path.join(ref_path, "cam00.png"))), 
                                              np.array(Image.open(os.path.join(ref_path, "cam01.png"))))


    depth_path = os.path.join(ref_path, "depthL.png")
    Image.fromarray(depth_to_saveable(depthL)).save(depth_path)
    cameras.add_depth(depth_path)

    depth_path = os.path.join(ref_path, "depthR.png")
    Image.fromarray(depth_to_saveable(depthR)).save(depth_path)
    cameras.add_depth(depth_path)

    augmentor = DataAugmentor(fov = args.fov)

    # Check what depth scale works best
    pose = cameras.create_pose([0,0,0], [args.baseline,0,0])

    # Move left image to match right image
    # Check different depth scales
    #1.9 best
    for factor in range(5, 50,1):
        augmentor.augment_single_image(cameras.get_imgs()[0], cameras.get_imgs()[1], cameras.get_depths()[0], pose, factor=factor/10)

if __name__ == '__main__':
    main()