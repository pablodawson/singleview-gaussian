import ffmpeg
import os

def extract_from_video(video_path, start_time=0, duration=2, cameras=None, output_path='output'):

    input_video = ffmpeg.input(video_path)
    
    print('Creating L/R views')

    left_half= input_video.filter("crop", "iw/2", "ih", 0, 0)
    right_half = input_video.filter("crop", "iw/2", "ih", "iw/2", 0)
    
    j = 0
    
    rectified = left_half.filter("v360", input="hequirect", 
                        output="rectilinear", v_fov=cameras.fov, h_fov =cameras.fov, w=cameras.res, h=cameras.res,
                        yaw=0, pitch=0)
    
    print(f'Creating cam{str(j).zfill(2)} frames')
    os.makedirs(f'{output_path}/cam{str(j).zfill(2)}', exist_ok=True)
    ffmpeg.output(rectified, os.path.join(output_path, f'cam{str(j).zfill(2)}/%04d.png'), t=duration, loglevel="error").run()
    
    j = 1

    rectified = right_half.filter("v360", input="hequirect", 
                        output="rectilinear", v_fov=cameras.fov, h_fov =cameras.fov, w=cameras.res, h=cameras.res,
                        yaw=0, pitch=0)
    
    print(f'Creating cam{str(j).zfill(2)} frames')
    os.makedirs(f'{output_path}/cam{str(j).zfill(2)}', exist_ok=True)
    ffmpeg.output(rectified, os.path.join(output_path, f'cam{str(j).zfill(2)}/%04d.png'), t=duration, loglevel="error").run()
    
    print('Done extraction')