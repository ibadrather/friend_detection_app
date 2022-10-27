from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort
import numpy as np
import face_recognition
import cv2
from utils import crop_image
from annotation_prediction_utils import face_image_for_onnx_model, MyRec
from starlette.responses import Response


# Loading trained Model
ort_session = ort.InferenceSession("models/my_friend_detection_v2.onnx")

# Encoding
encoding_ = np.load("models/class_encoding.npy")

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

@app.get("/")
def index():
    return {"status": "ok"}

@app.post("/upload_image")
async def receive_image(img: UploadFile=File(...)):
    ## Receiving and decoding the image
    contents = await img.read()

    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resizing without changing aspect ratio
    h, w, c = cv2_img.shape

    aspect_ratio = w/h

    image = cv2.resize(cv2_img, (720, int(720/aspect_ratio)), interpolation=cv2.INTER_AREA)

    # Finding face location
    face_locations = face_recognition.face_locations(image)

    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = crop_image(image, (top, left, bottom, right))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)

        # prepare for NN inference
        face_image = face_image_for_onnx_model(face_image)

        # get model prediction
        ort_inputs = {ort_session.get_inputs()[0].name: face_image}
        prediction = ort_session.run(None, ort_inputs)[0].argmax()

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

    ### Encoding and responding with the image
    captioned_image = cv2.imencode('.png', image)[1] # extension depends on which format is sent from Streamlit
    return Response(content=captioned_image.tobytes(), media_type="image/png")
