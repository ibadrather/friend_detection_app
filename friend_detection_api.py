from fastapi import FastAPI, File, UploadFile
import io
from starlette.responses import StreamingResponse
import cv2

app = FastAPI()


@app.post("/files/")
async def create_file(file: bytes = File()):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}


@app.post("/upload_image/")
def image_endpoint(*, file: UploadFile):
    # Returns a cv2 image array from the document vector
    print("fileeeeeeeee", file.filename)
    image = cv2.imread(file.file)
    res, im_png = cv2.imencode(".png", image)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
