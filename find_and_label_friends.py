import cv2
import face_recognition
import os.path as osp
import os
from utils import make_folder, crop_image
from tqdm import tqdm
import onnxruntime as ort
from annotation_prediction_utils import face_image_for_onnx_model, MyRec
import numpy as np
try:
    os.system("clear")
except:
    pass

# Loading trained Model
ort_session = ort.InferenceSession("my_friend_detection_v2.onnx")


# Images for testing
test_images_dir = "/home/ibad/Desktop/friend_detection_app/test_images/"
test_images = [
    osp.join(test_images_dir, image)
    for image in os.listdir(test_images_dir)
    if ".jpg" in image or ".png" in image
][:]

# Encoding
encoding_ = np.load("/home/ibad/Desktop/friend_detection_app/classes_ecoding.npy")

# font
font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 1
# Blue color in BGR
color = (0, 0, 255)
# Line thickness of 2 px
thickness = 2

# print(test_images)
make_folder("test_results")
for test_image in tqdm(test_images, leave=False):
    image = face_recognition.load_image_file(test_image)

    h, w, c = image.shape
    image = cv2.resize(image, (w // 6, h // 6), interpolation=cv2.INTER_AREA)

    face_locations = face_recognition.face_locations(image)
    for face_location in tqdm(face_locations):
        top, right, bottom, left = face_location
        face_image = crop_image(image, (top, left, bottom, right))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # prepare for NN inference
        face_image = face_image_for_onnx_model(face_image)

        # get model prediction
        ort_inputs = {ort_session.get_inputs()[0].name: face_image}
        prediction = ort_session.run(None, ort_inputs)[0].argmax()

        # if prediction == 0:
        #     prediction = "Murad"
        # elif prediction == 1:
        #     prediction = "Ibad"
        # elif prediction == 2:
        #     prediction = "Adnan"

        prediction = encoding_[prediction]

        # Draw rectangle and write predicted label
        cv2.rectangle(image, (left, top), (right, bottom), (220, 255, 220), 1)
        MyRec(image, left, top, right - left, bottom - top, 10, (0, 250, 0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            image,
            prediction,
            (left, top),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Image", image)
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()

    # Save image
    cv2.imwrite(osp.join("test_results", osp.basename(test_image)), image)

print("All images annotated and saved in test_results folder")
