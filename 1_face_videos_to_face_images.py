"""
    This script converts a video file to frames and saves them in a folder.
    We can also rotate the frames and change the frame rate.
"""
import os
import os.path as osp
from utils import video_to_frames, make_folder
from tqdm import tqdm

video_list = [
    (osp.join("videos", video), video.split("_")[0])
    for video in os.listdir("videos")
    if ".mp4" in video
    or ".avi" in video
    or ".mov" in video
    or ".mkv" in video
    or ".webm" in video
    or ".flv" in video
    or ".wmv" in video
]

# get frame from video
# we will not take all frames, we will take every 15th frame
for video in tqdm(video_list):
    frame_dir = osp.join("dataset", video[1])
    make_folder(frame_dir)
    video_to_frames(video[0], frame_dir, frame_rate=5, rotate=180)
