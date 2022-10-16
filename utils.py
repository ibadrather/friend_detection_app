import numpy as np
import cv2
import os.path as osp
import os


def video_to_frames(
    video_path, frames_path, frame_rate=1, name="frame", ext="jpg", rotate=0
):
    """
    Extract frames from a video and save them as images.
    :param video_path: Path to the video file.
    :param frames_path: Path to the folder where the frames will be saved.
    :param frame_rate: Frame rate to extract frames.
    :param name: Name of the frames.
    :param ext: Extension of the frames.
    :param rotate: Rotate the frames.
    :return: None
    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    count_n = 0
    h, w, c = image.shape
    while success:
        if count % frame_rate == 0:
            if rotate == 180:
                image = np.rot90(image, 2)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(
                osp.join(frames_path, f"{name}_{count_n}.png"), image
            )  # save frame as JPEG file
            count_n += 1
        success, image = vidcap.read()
        count += 1

    return


def make_folder(path):
    """
    Create a folder if it doesn't exist.
    :param path: Path to the folder.
    :return: None
    """
    if not osp.exists(path):
        os.makedirs(path)

    return


def crop_image(image, box):
    return image[box[0] : box[2], box[1] : box[3]]
