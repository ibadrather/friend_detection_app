from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort
import numpy as np
import face_recognition
import cv2
from utils import make_folder, crop_image
from annotation_prediction_utils import face_image_for_onnx_model, MyRec
import io
from starlette.responses import StreamingResponse


# Loading trained Model
ort_session = ort.InferenceSession("my_friend_detection_v2.onnx")

# Encoding
encoding_ = np.load("classes_encoding.npy")

# font
font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 1
# Blue color in BGR
color = (0, 0, 255)
# Line thickness of 2 px
thickness = 2


# API
app = FastAPI()

@app.post("/")
async def main(image: UploadFile = File(...)):
    image_typesbytes = await image.read()
    nparr = np.fromstring(image_typesbytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    h, w, c = img_np.shape
    img_np = cv2.resize(img_np, (w // 6, h // 6), interpolation=cv2.INTER_AREA)

    # Finding face location
    # print("Finding face location")
    face_locations = face_recognition.face_locations(img_np)
    # print("Working.......")
    # print(face_locations)
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = crop_image(img_np, (top, left, bottom, right))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)

        # prepare for NN inference
        face_image = face_image_for_onnx_model(face_image)

        # get model prediction
        ort_inputs = {ort_session.get_inputs()[0].name: face_image}
        prediction = ort_session.run(None, ort_inputs)[0].argmax()

        prediction = encoding_[prediction]

        # print(prediction)

        # Draw rectangle and write predicted label
        cv2.rectangle(img_np, (left, top), (right, bottom), (220, 255, 220), 1)
        MyRec(img_np, left, top, right - left, bottom - top, 10, (0, 250, 0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            img_np,
            prediction,
            (left, top),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    #cv2.imwrite("result.jpg", img_np)

    return StreamingResponse(io.BytesIO(img_np.tobytes()), media_type="image/png")