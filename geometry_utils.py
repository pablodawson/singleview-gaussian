import cv2
import numpy as np
import os
import json
import shutil
import glob
from geometry import depth_to_points
import math
import transforms3d
from scipy.spatial.transform import Rotation as R

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

class Cameras:
    def __init__(self, fov=80, res=1080, baseline=0.068):
        self.fov = fov
        self.res = res
        self.fx = res/(2*np.tan(np.radians(fov)/2))
        self.fy = res/(2*np.tan(np.radians(fov)/2))
        self.fovX = np.radians(fov)
        self.fovY = np.radians(fov)

        self.K = np.array([[self.fx, 0, res/2],
                            [0, self.fy, res/2],
                            [0, 0, 1]])
        self.poses = []
        self.imgs = []
        self.depths = []
        self.masks = []
    
    def get_imgs(self):
        return self.imgs

    def add_img(self, img):
        self.imgs.append(img)
    
    def add_mask(self, mask):
        self.masks.append(mask)
    
    def add_depth(self, depth):
        self.depths.append(depth)

    def get_masks(self):
        return self.masks
    
    def get_depths(self):
        return self.depths

    def get_poses(self):
        return self.poses
    
    def create_pose(self, R_euler, T):
        x, y, z = T
        pose = np.eye(4)
        pose[0, 3] = x
        pose[1, 3] = y
        pose[2, 3] = z
        
        r = R.from_euler('xyz', R_euler)
        pose[:3, :3] = r.as_matrix()

        return pose


    def create_poses(self, theta=np.pi/8, offset = 0.12):

        # --- Augmented poses ---
        pose = self.create_pose([0, 0, 0], [0, 0, 0])
        self.poses.append(pose)

        pose = self.create_pose([0, -theta/2, 0], [-offset/2, 0, 0])
        self.poses.append(pose)

        pose = self.create_pose([0, theta/2, 0], [offset/2, 0, 0])
        self.poses.append(pose)

        pose = self.create_pose([0, -theta, 0], [-offset, 0, 0])
        self.poses.append(pose)

        pose = self.create_pose([0, theta, 0], [offset, 0, 0])
        self.poses.append(pose)
        

    def get_intrinsics(self):
        return self.K


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec
        
def unproject(image, depth, transform = None, iK = [1,1]):

    # Normalized screen coords
    x_grid = 2* np.arange(0, image.shape[1])/image.shape[1]-1
    y_grid = 2* np.arange(0, image.shape[0])/image.shape[0]-1
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)

    # Reproject
    depth_metric = depth
    z = - depth_metric
    x = (x_grid/iK[0]) * depth_metric
    y = -(y_grid/iK[1]) * depth_metric
    h = np.ones_like(x)

    # Stack
    points = np.stack([x,y,z,h], axis=-1)

    # Transform
    #points = np.matmul(points, transform)

    return points

def create_custom_dataset(points, colors, cameras, video_path):
    output_path = "dataset"
    os.makedirs(output_path, exist_ok=True)
    points_path = os.path.join(output_path, "points.npy")
    colors_path = os.path.join(output_path, "colors.npy")

    np.save(points_path, points)
    np.save(colors_path, colors)

    intrinsics_path = os.path.join(output_path, "intrinsics.json")
    intrinsics = {"FovX": cameras.fovX, "FovY": cameras.fovY}
    with open(intrinsics_path, 'w') as f:
        json.dump(intrinsics, f)

    poses = cameras.get_poses()
    poses_path = os.path.join(output_path, "poses.json")
    with open(poses_path, 'w') as f:
        json.dump(poses, f)

    input_images_folder = os.path.join(output_path, "input")
    os.makedirs(input_images_folder, exist_ok=True)
    input_images = glob.glob("input/cam*.png")
    for i, image in enumerate(input_images):
        shutil.copy(image, os.path.join(input_images_folder, f"{i}.png"))

    print("Custom dataset created")

def dummy_reconstruction(cameras):
    poses = cameras.get_poses()
    points = []
    colors = []
    for pose in poses:
        points.append(pose["pose"][:3, 3])
        colors.append([255, 0, 0])
    
    points = np.array(points)
    colors = np.array(colors)
    
    points[:, 0] *= -1
    #points[:, 1] *= -1
    #points[:, 2] *= -1

    return points, colors

# Creates point cloud
def get_reconstruction(colorL, depthL, cameras, mask=None, max_points = 20000):

    res = 256
    depth = cv2.resize(depthL, (res, res))
    color = cv2.resize(colorL, (res, res))
    fov = cameras.fov
    f = res/(2*np.tan(np.radians(fov)/2))

    K = np.array([[f, 0, res/2],
                    [0, f, res/2],
                    [0, 0, 1]])
    
    points3d = depth_to_points(depth[None], K = K)
    points3d = points3d.reshape(-1, 3)
    colors = color.reshape(-1, 3)

    #mask = points3d[:, 2] < 3.5
    if mask is not None:
        mask = cv2.resize(mask, (res, res))
        mask = (mask > 0).reshape(-1)
        points3d = points3d[mask]
        colors = colors[mask]
    
    if points3d.shape[0] > max_points:
        mask = np.random.choice(points3d.shape[0], max_points, replace=False)
        points3d = points3d[mask]
        colors = colors[mask]

    
    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    points3d[:, 1] *= -1
    points3d[:, 2] *= -1

    return points3d, colors

if __name__=="__main__":
    img = cv2.imread("color/1.png")
    depth = cv2.imread("depth/1.png")