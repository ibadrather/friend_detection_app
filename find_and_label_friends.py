import cv2
import face_recognition
import os.path as osp
import os
from utils import make_folder, crop_image
from tqdm import tqdm
import onnxruntime as ort


ort_session = ort.InferenceSession("my_friend_detection.onnx")

try:
    os.system("clear")
except:
    pass

def face_image_for_onnx_model(image):
    # image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = image.transpose((2, 0, 1))
    # Normalising the image here
    image = image / 255.0
    image = image.astype("float32")
    image = image.reshape(1, 3, 224, 224)
    return image

def MyRec(rgb,x,y,w,h,v=20,color=(200,0,0),thikness =2):
    """To draw stylish rectangle around the objects"""
    cv2.line(rgb, (x,y),(x+v,y), color, thikness)
    cv2.line(rgb, (x,y),(x,y+v), color, thikness)

    cv2.line(rgb, (x+w,y),(x+w-v,y), color, thikness)
    cv2.line(rgb, (x+w,y),(x+w,y+v), color, thikness)

    cv2.line(rgb, (x,y+h),(x,y+h-v), color, thikness)
    cv2.line(rgb, (x,y+h),(x+v,y+h), color, thikness)

    cv2.line(rgb, (x+w,y+h),(x+w,y+h-v), color, thikness)
    cv2.line(rgb, (x+w,y+h),(x+w-v,y+h), color, thikness)

# Images for testing
test_images_dir = "/home/ibad/Desktop/friend_detection_app/test_images/"
test_images = [
    osp.join(test_images_dir, image)
    for image in os.listdir(test_images_dir)
    if ".jpg" in image or ".png" in image
][:]


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
    image = cv2.resize(image, (w//6, h//6), interpolation=cv2.INTER_AREA)

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

        if prediction == 0:
            prediction = "Murad"
        elif prediction == 1:
            prediction = "Ibad"
        elif prediction == 2:
            prediction = "Adnan"
        
        print("\n Predicted", prediction)

        # face_image = (face_image*255)
        # face_image = face_image.squeeze(0)
        # face_image = face_image.transpose((1, 2, 0)).astype("uint8")
        # face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        # print(face_image.shape)
        # cv2.imshow("Face", face_image)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()

        # Now draw a rectangle around the face
        # cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        #print(top, right, bottom, left)

        x1, y1 = left, top
        x2, y2 = right, bottom

        #print(x1, y1, x2, y2)

        cv2.rectangle(image,(x1,y1),(x2,y2),(220,255,220),1)
        MyRec(image, x1, y1, x2 - x1, y2 - y1, 10, (0,250,0), 3)
        # save(gray,new_path+str(counter),(x1-fit,y1-fit,x2+fit,y2+fit))
        #save(gray,new_path+str(counter),(x1,y1,x2,y2))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, prediction, ((right+left)//2, (top+bottom)//2), font, fontScale, color, thickness, cv2.LINE_AA)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", image)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    break

        


