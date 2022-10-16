import cv2
import os
import os.path as osp
from utils import video_to_frames, make_folder
from tqdm import tqdm

video_list = [
    (osp.join("videos", video), video[:-4])
    for video in os.listdir("videos")
    if ".mp4" in video
]

# get frame from video
# we will not take all frames, we will take every 15th frame
for video in tqdm(video_list):
    frame_dir = osp.join("dataset", video[1])
    make_folder(frame_dir)
    video_to_frames(video[0], frame_dir, frame_rate=15, rotate=180)
