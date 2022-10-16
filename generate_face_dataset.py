import face_recognition
import cv2
import os.path as osp
import os
from utils import make_folder, crop_image
from tqdm import tqdm

base_path = "/home/ibad/Desktop/friend_detection_app/"

people = [
    osp.join("dataset", person)
    for person in os.listdir("dataset")
    if osp.isdir(osp.join("dataset", person))
]

# Get the faces for each person from the photographs and save them
for person in tqdm(people):
    person_dir = osp.join(base_path, "face_dataset", person.split("/")[-1])
    make_folder(person_dir)
    face_counter = 0
    for image in tqdm(os.listdir(person)):
        image_path = osp.join(person, image)
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = crop_image(image, (top, left, bottom, right))
            cv2.imwrite(osp.join(person_dir, str(face_counter) + ".png"), face_image)
            face_counter += 1
