Detect if my friends are in a photo.

### Steps involved.

1. Convert the video in which my friends into frames(images) at a certain frame rate (all frame data will be redundant).

2. Find and crop faces from the images and save the face crops in a different folder based on the friends (label) while dividing into training and validation sets.

3. Train a Neural Network on the face data generated and ensure its performance.

4. Convert the model to ONNX runtime format for inference in application.

5. Find faces in an image and label them accordingly.
