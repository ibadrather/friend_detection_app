import cv2
import os
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset
from sklearn import preprocessing  # for label encoding


class FaceDataset(Dataset):
    def __init__(self, data_dir):
        people = [
            osp.join(data_dir, person)
            for person in os.listdir(data_dir)
            if osp.isdir(osp.join(data_dir, person))
        ]
        self.all_data = []

        for person in people:
            data = [
                (osp.join(person, photo), person.split("/")[-1])
                for photo in os.listdir(person)
                if ".png" in photo or ".jpg" in photo
            ]

            self.all_data.extend(data)

        # Generating Labels
        self.labels = [label for _, label in self.all_data]

        # Label Encoding
        le = preprocessing.LabelEncoder()
        le.fit_transform(self.labels)

        self.labels = le.classes_

        # Save encoding data
        np.save("class_encoding", le.classes_)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        image_path, label = self.all_data[idx]

        label = np.where(self.labels == label)[0]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = image.transpose((2, 0, 1))
        # Normalising the image here
        image = image / 255.0
        image = torch.from_numpy(image).float()
        return image, label


def main():
    from torch.utils.data import DataLoader

    try:
        os.system("clear")
    except:
        pass

    train_dataset = FaceDataset(osp.join("face_dataset", "train"))
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # print(np.load("/home/ibad/Desktop/friend_detection_app/classes_ecoding.npy"))

    a = iter(train_dataloader)
    c, d = next(a)

    print(d.shape)


if __name__ == "__main__":
    main()
