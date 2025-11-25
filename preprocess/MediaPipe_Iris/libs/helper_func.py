import cv2
import os
from tqdm import tqdm
from PIL import Image

def vid2images(video_path, out_path):
    video_cap = cv2.VideoCapture(video_path)
    length = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for i in tqdm(range(length)):
        ret, frame = video_cap.read()

        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(out_path+'/{:05d}.png'.format(i))
        if ret == False:
            break

def images2vid(source_folder, output_dir, framerate=30, image_type='png'):
    os.system("ffmpeg -framerate {} -pattern_type glob -i '{}/*.{}' -c:v libx264 -pix_fmt yuv420p {}/video.mp4".format(framerate, source_folder, image_type, output_dir))