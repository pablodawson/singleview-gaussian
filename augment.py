import os
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from PIL import Image
import pyrender
from inpaint import Inpainter
from foreground_segmentation import estimate_foreground_mask
from space_filler import SpaceFiller

from geometry_utils import get_reconstruction
from geometry import get_mesh

from rembg import remove
# Modificar scene_info para incluir depths -> dataset devuelve camera con estimated_depth

class DataAugmentor:
    
    def __init__(self, fov=80, adj_size = (512, 512)):

        self.size = adj_size
        self.scene = pyrender.Scene(ambient_light=[1., 1., 1.])
        self.flags = pyrender.constants.RenderFlags.FLAT | pyrender.constants.RenderFlags.SKIP_CULL_FACES
        self.fov = fov
        self.camera = pyrender.PerspectiveCamera(yfov=np.radians(fov), aspectRatio=adj_size[0]/adj_size[1], znear=0.001)
        
        camera_pose = np.eye(4)
        self.scene.add(self.camera, pose=camera_pose)
        
        self.renderer = pyrender.OffscreenRenderer(adj_size[0], adj_size[1])

        self.inpainter = Inpainter()
        self.spacefiller = SpaceFiller()

        self.points = None
        self.colors = None

    def get_points(self):
        return self.points, self.colors
    
    def depth_to_saveable(self, depth):
        return (depth * (2**13 - 1)).astype(np.uint16)
    
    def create_mask(self, img):
        if type(img) == np.array:
            return cv2.inRange(img, (255, 255, 255), (255,255, 255))
        elif type(img) == torch.Tensor:
            mask = 1.0 - (img == 1.0).all(dim=1).unsqueeze(0).float()
            return mask

    def c2p_axis(self, point):
        point_new = point
        point_new[0]*=-1
        return point_new

    
    def augment_images(self, cameras, output_path= "output"):

        image_init = Image.open(os.path.join(output_path, "init_image.png"))
        depth_init = np.array(Image.open(os.path.join(output_path, "init_depth.png")))/ (2**13 - 1)
        
        f = self.size[0]/(2*np.tan(np.radians(self.fov)/2))
        K = np.array([[f, 0, self.size[0]/2],
                        [0, f, self.size[0]/2],
                        [0, 0, 1]])
        
        # Get foreground mask, inpaint to get background
        #dispL = 1.0 / (depth_init + 1e-6)
        #dispL = (dispL - np.percentile(dispL,10)) / (np.percentile(dispL,90) - np.percentile(dispL,10))
        color_img = image_init.resize(self.size)
        #disp_img = (cv2.resize(dispL, self.size) * 255).astype(np.uint8)

        bg = remove(image_init)
        bg = np.array(bg)
        bg = cv2.resize(bg, self.size)
        f_mask = np.where(bg[:, :, 3] > 0, 255, 0).astype(np.uint8)

        f_mask_expanded = cv2.dilate(f_mask, np.ones((5,5),np.uint8), iterations = 3)
        
        background_image = self.inpainter(color_img, f_mask_expanded, strength=1.0)
        background_depth = cv2.resize(depth_init, self.size)

        background_depth = self.spacefiller(background_depth, background_image, (f_mask_expanded.copy()/255.0).astype(float))

        cv2.imwrite(os.path.join(output_path,"background.png"), background_image)
        Image.fromarray(self.depth_to_saveable(background_depth)).save(os.path.join(output_path,"background_depth.png"))
        cv2.imwrite(os.path.join(output_path,"foreground_mask.png"), f_mask)

        # Background reconstruction
        background_image = Image.fromarray(background_image).resize(self.size)
        background_depth = cv2.resize(background_depth, self.size)
        reconstruction = get_mesh(background_image, background_depth, K=K)
        mesh = pyrender.Mesh.from_trimesh(reconstruction)
        B_mesh_node = pyrender.Node(mesh=mesh, matrix=np.eye(4))

        # Start with those points
        self.points, self.colors = get_reconstruction(np.array(background_image), background_depth, cameras, max_points=20000)

        # Foreground reconstruction
        color_img = image_init.resize(self.size)
        depth_img = cv2.resize(depth_init, self.size)

        reconstruction = get_mesh(color_img, depth_img, K=K, mask=f_mask)
        mesh = pyrender.Mesh.from_trimesh(reconstruction)
        F_mesh_node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
        self.scene.add_node(F_mesh_node)
        foreground_color, foreground_depth = self.renderer.render(self.scene, flags= self.flags)
        cv2.imwrite(f"foreground_image.png", foreground_color)

        ###

        # Add points
        points, colors = get_reconstruction(np.array(color_img), depth_img, cameras, mask=f_mask, max_points=10000)
        self.points = np.concatenate((self.points, points), axis=0)
        self.colors = np.concatenate((self.colors, colors), axis=0)
        
        poses = cameras.get_poses()
        
        # Add background
        self.scene.add_node(B_mesh_node)

        img_aug_path = os.path.join(output_path, "images_aug")
        depth_aug_path = os.path.join(output_path, "depths_aug")

        os.makedirs(img_aug_path, exist_ok=True)
        os.makedirs(depth_aug_path, exist_ok=True)

        # Augment images
        for i, pose in enumerate(poses):
            
            campose = pose.copy()
            
            self.scene.set_pose(self.scene.main_camera_node, campose)

            # Render full scene
            color, depth = self.renderer.render(self.scene, flags= self.flags)

            # Render foreground only
            self.scene.remove_node(B_mesh_node)
            foreground_color, foreground_depth = self.renderer.render(self.scene, flags= self.flags)
            self.scene.add_node(B_mesh_node)

            cv2.imwrite(f"{i}.png", color)
            Image.fromarray(self.depth_to_saveable(depth)).save(f"{i}_depth.png")
            cv2.imwrite(f"{i}_foreground.png", foreground_color)
            cv2.imwrite(f"{i}_foreground_depth.png", foreground_depth)
            
            # Create mask

            # Holes in the main image
            mask_init = cv2.inRange(color, (255,255,255), (255,255,255))
            mask_init = cv2.dilate(mask_init, np.ones((5,5), np.uint8))

            # Foreground expansion
            foreground_inv = cv2.inRange(foreground_color, (255,255,255), (255,255,255))
            foreground = 255 - foreground_inv
            foreground_expanded = cv2.dilate(foreground, np.ones((5,5), np.uint8), iterations=4)

            expansion = cv2.bitwise_and(foreground_expanded, foreground_inv)
            expansion = cv2.dilate(expansion, np.ones((5,5), np.uint8))

            # Blend the two
            mask = cv2.bitwise_or(expansion, mask_init).copy()

            col = cv2.resize(color, (512, 512))
            color = self.inpainter(col.copy(), mask.copy())
            color = cv2.resize(color, self.size)
            
            depth_fill = self.spacefiller(depth, color, (mask/255.0).astype(float))
            depth_fill = cv2.resize(depth_fill, (self.size))

            Image.fromarray(self.depth_to_saveable(depth_fill)).save(os.path.join(depth_aug_path, f"{i}.png"))
            Image.fromarray(color).save(os.path.join(img_aug_path, f"{i}.png"))
            
            cameras.add_img(os.path.join(img_aug_path, f"{i}.png"))
            cameras.add_depth(os.path.join(depth_aug_path, f"{i}.png"))
            
            cv2.imwrite("expanded_foreground.png", expansion)
            cv2.imwrite("holes.png", mask_init)
            cv2.imwrite("foreground.png", foreground)
            cv2.imwrite("full_mask.png", mask)
            

            # Add points
            points, colors = get_reconstruction(color, depth_fill, cameras, mask=mask, max_points=5000)
            h_points = np.hstack((points, np.ones((len(points), 1))))
            transformed_points = np.dot(campose, h_points.T).T[:, :3]
            points = transformed_points

            self.points = np.concatenate((self.points, points), axis=0)
            self.colors = np.concatenate((self.colors, colors), axis=0)
            
            # Add mesh
            # Needs in relation to the init position ? 
            reconstruction = get_mesh(color, depth_fill, K=K, mask=mask)
            mesh = pyrender.Mesh.from_trimesh(reconstruction)
            Aug_mesh_node = pyrender.Node(mesh=mesh, matrix=campose)
            self.scene.add_node(Aug_mesh_node)

            # Render with augmented mesh
            color, depth = self.renderer.render(self.scene, flags= self.flags)
            cv2.imwrite(f"{i}_aug_newmesh.png", color)
            Image.fromarray(self.depth_to_saveable(depth)).save(f"{i}_aug_newmesh_depth.png")

            