import numpy as np
import os
import glob
from pre_colmap import COLMAPDatabase
from geometry_utils import rotmat2qvec
import shutil
from scipy.spatial.transform import Rotation as R

def flip_rotation_axes(rotation_matrix, flip_x=False, flip_y=False, flip_z=False):
    """
    Flip the specified axes of a 3x3 rotation matrix.

    Args:
    rotation_matrix (np.array): The original 3x3 rotation matrix.
    flip_x (bool): Whether to flip the X-axis.
    flip_y (bool): Whether to flip the Y-axis.
    flip_z (bool): Whether to flip the Z-axis.

    Returns:
    np.array: The rotation matrix after flipping the specified axes.
    """
    flipped_matrix = rotation_matrix.copy()

    if flip_x:
        flipped_matrix[1:3, :] = -flipped_matrix[1:3, :]

    if flip_y:
        flipped_matrix[[0, 2], :] = -flipped_matrix[[0, 2], :]

    if flip_z:
        flipped_matrix[:, [0, 1]] = -flipped_matrix[:, [0, 1]]

    return flipped_matrix

def flip_translation_vector(translation_vector, flip_x=False, flip_y=False, flip_z=False):
    """
    Flip the specified axes of a translation vector.

    Args:
    translation_vector (np.array): The original translation vector.
    flip_x (bool): Whether to flip along the X-axis.
    flip_y (bool): Whether to flip along the Y-axis.
    flip_z (bool): Whether to flip along the Z-axis.

    Returns:
    np.array: The translation vector after flipping the specified axes.
    """
    flipped_vector = translation_vector.copy()

    if flip_x:
        flipped_vector[0] = -flipped_vector[0]

    if flip_y:
        flipped_vector[1] = -flipped_vector[1]

    if flip_z:
        flipped_vector[2] = -flipped_vector[2]

    return flipped_vector

def transform_to_colmap(rot_matrix, trans_vector):
    # Transformation matrix from your system to Colmap's system
    transform = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    
    rot_matrix_colmap = transform @ rot_matrix @ transform.T
    trans_vector_colmap = transform @ rot_matrix @ trans_vector
    
    return rot_matrix_colmap, trans_vector_colmap

def create_colmap_dataset(path, points3d, colors, cameras, offset=0, frame="0001.png"):
    
    projectfolder = os.path.join(path, "colmap_" + str(offset))
    outfolder = os.path.join(projectfolder, "sparse")
    os.makedirs(outfolder, exist_ok=True)
    outfolder = os.path.join(outfolder, "0")
    os.makedirs(outfolder, exist_ok=True)

    savetxt = os.path.join(outfolder, "images.txt")
    savecamera = os.path.join(outfolder, "cameras.txt")
    savepoints = os.path.join(outfolder, "points3D.txt")
    imagetxtlist = []
    cameratxtlist = []
    pointtxtlist = []
    if os.path.exists(os.path.join(projectfolder, "input.db")):
        os.remove(os.path.join(projectfolder, "input.db"))

    db = COLMAPDatabase.connect(os.path.join(projectfolder, "input.db"))
    
    db.create_tables()
    
    points3d *= -1

    # Add points
    pointtxtlist.append("# 3D point list with one line of data per point:\n")
    pointtxtlist.append("#  POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    pointtxtlist.append("# Number of points: " + str(points3d.shape[0]) + ", mean track length: 4.3428430485899856\n")
    
    for i in range(points3d.shape[0]):
        point = points3d[i]
        color = colors[i]
        pointtxtlist.append(str(i+1) + " " + " ".join([str(x) for x in point]) + " " + " ".join([str(x) for x in color]) + " 0.95230286670052422 10 10 10 10 10 10\n")
    
    with open(savepoints, "w") as f:
        f.writelines(pointtxtlist)

    w2c_matriclist = cameras.poses
    
    for i in range(len(w2c_matriclist)):
        cameraname = os.path.basename("cam" + str(i).zfill(2))
        m =w2c_matriclist[i]
                
        colmapT = m[:3, 3]
        colmapR = m[:3, :3]

        colmapR, colmapT = transform_to_colmap(colmapR, colmapT)
        
        H = cameras.res
        W = cameras.res
        focal = cameras.fx
        
        colmapQ = rotmat2qvec(colmapR)

        imageid = str(i+1)
        cameraid = imageid
        pngname = os.path.basename(cameras.get_imgs()[i])
        
        line =  imageid + " "

        for j in range(4):
            line += str(colmapQ[j]) + " "
        for j in range(3):
            line += str(colmapT[j]) + " "
        line = line  + cameraid + " " + pngname + "\n"
        empltyline = "\n"
        imagetxtlist.append(line)
        imagetxtlist.append(empltyline)

        focolength = focal
        model, width, height, params = i, W, H, np.array((focolength,  focolength, W//2, H//2,))

        camera_id = db.add_camera(1, width, height, params)
        cameraline = str(i+1) + " " + "PINHOLE " + str(width) +  " " + str(height) + " " + str(focolength) + " " + str(focolength) + " " + str(W//2) + " " + str(H//2) + "\n"
        cameratxtlist.append(cameraline)
        
        image_id = db.add_image(pngname, camera_id,  prior_q=np.array((colmapQ[0], colmapQ[1], colmapQ[2], colmapQ[3])), prior_t=np.array((colmapT[0], colmapT[1], colmapT[2])), image_id=i+1)

    with open(savetxt, "w") as f:
        f.writelines(imagetxtlist)
    with open(savecamera, "w") as f:
        f.writelines(cameratxtlist)
    
     # --- Extract cams of current frames ---
    images_path = os.path.join(projectfolder, 'images')
    os.makedirs(images_path, exist_ok=True)

    folders = [f for f in os.listdir(path)]
    
    #for folder in folders:
     #   if folder.startswith('cam'):
    #        shutil.copy(f'{path}/{folder}/{frame}', f'{images_path}/{folder}.png')
    for image in cameras.get_imgs():
        shutil.copy(image, os.path.join(images_path, os.path.basename(image)))
    
    db.commit()
    db.close()