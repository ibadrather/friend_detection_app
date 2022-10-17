import face_recognition
import cv2
import os.path as osp
import os
from utils import make_folder, crop_image
from tqdm import tqdm
import random
import time

base_path = "/home/ibad/Desktop/friend_detection_app/"

people = [
    osp.join("dataset", person)
    for person in os.listdir("dataset")
    if osp.isdir(osp.join("dataset", person))
]

# Percentage of train samples
train_percent = 0.8

# Get the faces for each person from the photographs and save them
for person in tqdm(people):
    person_dir_train = osp.join(
        base_path, "face_dataset", "train", person.split("/")[-1]
    )
    person_dir_val = osp.join(base_path, "face_dataset", "val", person.split("/")[-1])

    make_folder(person_dir_train)
    make_folder(person_dir_val)
    # First we will get all images associated with a person and save they dir loc in a list
    all_images = []
    for image in tqdm(os.listdir(person)):
        image_path = osp.join(person, image)
        all_images.append(image_path)

    # Now we will split images into train and val
    random.shuffle(all_images)
    train_images = all_images[: int(train_percent * len(all_images))]
    val_images = all_images[int(train_percent * len(all_images)) :]

    # Save the faces in the train folder
    for image in tqdm(train_images):
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = crop_image(image, (top, left, bottom, right))
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(
                osp.join(person_dir_train, f"{str(int(time.time()*1e4))}.png"),
                face_image,
            )

    # Save the faces in the val folder
    for image in tqdm(val_images):
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = crop_image(image, (top, left, bottom, right))
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(
                osp.join(person_dir_val, f"{str(int(time.time()*1e4))}.png"), face_image
            )
