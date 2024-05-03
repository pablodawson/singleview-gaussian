import numpy as np
import os
import glob
from pre_colmap import COLMAPDatabase
from geometry_utils import rotmat2qvec
import shutil

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

    points3d[:,0] *= -1
    #points3d[:,1] *= -1

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
        m = w2c_matriclist[i]
        
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        m[:3, 1:3] *= -1

        colmapR = m[:3, :3]
        T = m[:3, 3]
        T[0] = -T[0]
        
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
            line += str(T[j]) + " "
        line = line  + cameraid + " " + pngname + "\n"
        empltyline = "\n"
        imagetxtlist.append(line)
        imagetxtlist.append(empltyline)

        focolength = focal
        model, width, height, params = i, W, H, np.array((focolength,  focolength, W//2, H//2,))

        camera_id = db.add_camera(1, width, height, params)
        cameraline = str(i+1) + " " + "PINHOLE " + str(width) +  " " + str(height) + " " + str(focolength) + " " + str(focolength) + " " + str(W//2) + " " + str(H//2) + "\n"
        cameratxtlist.append(cameraline)
        
        image_id = db.add_image(pngname, camera_id,  prior_q=np.array((colmapQ[0], colmapQ[1], colmapQ[2], colmapQ[3])), prior_t=np.array((T[0], T[1], T[2])), image_id=i+1)

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