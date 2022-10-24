import os
import torch
import numpy as np
import pandas as pd
from model import NetExample
from ptl_modules import FriendFaceDetector
from dataloading import FaceDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import onnxruntime as ort

try:
    os.system("clear")
except:
    pass

# Model Checkpoint
model_dir = "/home/ibad/Desktop/friend_detection_app/lightning_logs/friend-detection/version_9/checkpoints/epoch=43-step=3080.ckpt"

# Loading the state dict into the model
net = NetExample(in_channels=3, out_classes=4)
torch_model = FriendFaceDetector(net)
state_dict = torch.load(model_dir)["state_dict"]
torch_model.load_state_dict(state_dict)

print("Model Loaded Successfully")

# Model in Evaluation Mode
torch_model.eval()

# Load Test Data
test_data = FaceDataset("/home/ibad/Desktop/friend_detection_app/face_dataset/val")
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

# Getting Predictions for PyTorch Model
predicted_pt = []
actual_pt = []
for image, label in test_dataloader:
    output = torch_model(image)
    # print("Predicted: ", output.argmax().item(), "Actual: ", label.item())
    predicted_pt.append(output.argmax().item())
    actual_pt.append(label.item())


print(
    "Claasification Report (PyTorch): \n",
    classification_report(actual_pt, predicted_pt),
)

print("Accuracy Score (PyTorch): ", accuracy_score(actual_pt, predicted_pt))
print("Confusion Matrix: \n", confusion_matrix(actual_pt, predicted_pt))

# Now we will convert to ONNX
# Export the model
torch.onnx.export(
    torch_model,  # model being run
    image,  # model input (or a tuple for multiple inputs)
    "my_friend_detection.onnx",  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=10,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input"],  # the model's input names
    output_names=["output"],  # the model's output names
    dynamic_axes={
        "input": {0: "batch_size"},  # variable length axes
        "output": {0: "batch_size"},
    },
)

print("Model converted to ONNX")

# Let' test our converted model if accuracy is same

ort_session = ort.InferenceSession("my_friend_detection.onnx")


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


predicted_onnx = []
actual_onnx = []

for image, label in test_dataloader:
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image)}
    ort_outs = ort_session.run(None, ort_inputs)[0]

    # print("Predicted: ", output.argmax().item(), "Actual: ", label.item())
    predicted_onnx.append(ort_outs.argmax())
    actual_onnx.append(label.item())


print("Accuracy Score (ONNX): ", accuracy_score(actual_onnx, predicted_onnx))
print(
    "Claasification Report (ONNX): \n",
    classification_report(actual_onnx, predicted_onnx),
)
print("Confusion Matrix: \n", confusion_matrix(actual_onnx, predicted_onnx))
