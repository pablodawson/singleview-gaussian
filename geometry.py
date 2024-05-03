import numpy as np
import cv2
import trimesh
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def depth_edges_mask(depth):
    """Returns a mask of edges in the depth map.
    Args:
    depth: 2D numpy array of shape (H, W) with dtype float32.
    Returns:
    mask: 2D numpy array of shape (H, W) with dtype bool.
    """
    # Compute the x and y gradients of the depth map.
    depth_dx, depth_dy = np.gradient(depth)
    # Compute the gradient magnitude.
    depth_grad = np.sqrt(depth_dx ** 2 + depth_dy ** 2)
    # Compute the edge mask.
    mask = depth_grad > 0.3
    return mask

def create_triangles(h, w, mask=None):
    """Creates mesh triangle indices from a given pixel grid size.
        This function is not and need not be differentiable as triangle indices are
        fixed.
    Args:
    h: (int) denoting the height of the image.
    w: (int) denoting the width of the image.
    Returns:
    triangles: 2D numpy array of indices (int) with shape (2(W-1)(H-1) x 3)
    """
    x, y = np.meshgrid(range(w - 1), range(h - 1))
    tl = y * w + x
    tr = y * w + x + 1
    bl = (y + 1) * w + x
    br = (y + 1) * w + x + 1
    triangles = np.array([tl, bl, tr, br, tr, bl])
    triangles = np.transpose(triangles, (1, 2, 0)).reshape(
        ((w - 1) * (h - 1) * 2, 3))
    if mask is not None:
        mask = mask.reshape(-1)
        triangles = triangles[mask[triangles].all(1)]
    return triangles


def get_coords(image, depth, filter_baseline_area=True):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (depth.shape[0], depth.shape[1]))
    
    if len(depth.shape) == 2:
        width, height = depth.shape
        depth_normalized = depth.astype(np.float32)
    else:
        width, height, _ = depth.shape

        depth_normalized = depth[:,:,0].astype(np.float32)

    if filter_baseline_area:
        # Filter out the area around the baseline
        depth_normalized[:, :int(height/7)] = 0
        depth_normalized[:, int(6*height/7):] = 0

    distances = np.zeros((width, height), dtype=np.float32)
    distances = depth_normalized

    phi = np.linspace(0, np.pi, width)
    theta = np.linspace(0, np.pi, height)

    phi, theta = np.meshgrid(phi, theta)

    unit_half_sphere = np.array([np.sin(phi) * np.cos(theta), np.cos(phi),  np.sin(phi) * np.sin(theta)]).transpose(1, 2, 0)

    final_coords = np.zeros((width, height, 3), dtype=np.float32)
    final_coords = -unit_half_sphere * distances[..., np.newaxis] # volver a cambiar sin -

    return final_coords

def create_point_cloud(image, coords, baseline= 1, debug = False):
    colors = image.reshape(-1, 3)
    verts = coords[:,:,:3].reshape(-1, 3)

    # Make y down and z forward
    verts = np.array([verts[:, 1], -verts[:, 0], verts[:, 2]]).transpose(1,0)

    distances = np.linalg.norm(verts, axis=1)

    # remove outliers by distance
    lower_bound = np.percentile(distances, 20)
    upper_bound = np.percentile(distances, 95)

    mask = (distances > lower_bound) & (distances < upper_bound)

    verts = verts[mask]
    colors = colors[mask]

    # create a random selection
    mask = np.random.choice(verts.shape[0], 10_000, replace=False)
    verts = verts[mask]
    colors = colors[mask]

    if (debug):
        
        x = verts[:, 0]
        y = verts[:, 1]
        z = verts[:, 2]
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=colors/255, s=1)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.scatter(0, 0, 0, c='r', s=20)
        ax.scatter(baseline*2, 0, 0, c='g', s=20)
        plt.show()

    return verts, colors

def depth_to_points(depth, R=None, t=None, K = None):

    Kinv = np.linalg.inv(K)
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    # M converts from your coordinate to PyTorch3D's coordinate system
    M = np.eye(3)
    #M[0, 0] = -1.0
    #M[1, 1] = -1.0

    height, width = depth.shape[1:3]

    x = width - np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    # coord = torch.as_tensor(coord, dtype=torch.float32, device=device)
    coord = coord[None]  # bs, h, w, 3

    D = depth[:, :, :, None, None]
    # print(D.shape, Kinv[None, None, None, ...].shape, coord[:, :, :, :, None].shape )
    pts3D_1 = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
    # pts3D_1 live in your coordinate system. Convert them to Py3D's
    pts3D_1 = M[None, None, None, ...] @ pts3D_1
    # from reference to targe tviewpoint
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
    # pts3D_2 = pts3D_1
    # depth_2 = pts3D_2[:, :, :, 2, :]  # b,1,h,w
    return pts3D_2[:, :, :, :3, 0][0]


# Creates mesh
def get_mesh(image, depth, return_occlusions=False, save=False, K = None, mask=None):
    pts3d = -depth_to_points(depth[None], K = K)
    pts3d = pts3d.reshape(-1, 3)

    # Create a trimesh mesh from the points
    # Each pixel is connected to its 4 neighbors
    # colors are the RGB values of the image

    verts = pts3d.reshape(-1, 3)
    image = np.array(image)

    if return_occlusions:
        if mask is not None:
            mask = (mask>0) & depth_edges_mask(depth)
        else:
            mask = depth_edges_mask(depth)
        
        triangles = create_triangles(image.shape[0], image.shape[1], mask=mask)
        
        image_new = image.copy()
        image_new[mask] = [0,0,0]
        colors = image_new.reshape(-1, 3)

    else:
        if mask is not None:
            triang_mask = (mask > 0) & (~depth_edges_mask(depth))
        else:
            triang_mask = ~depth_edges_mask(depth)
        triangles = create_triangles(image.shape[0], image.shape[1], mask= triang_mask)
        colors = image.reshape(-1, 3)
    
    mesh = trimesh.Trimesh(vertices=verts, faces=triangles, vertex_colors=colors)
    if save:
        mesh.export("out.glb")
    
    return mesh