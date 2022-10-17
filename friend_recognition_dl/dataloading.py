import cv2
import os
import os.path as osp

import torch
from torch.utils.data import Dataset, DataLoader


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

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        image_path, label = self.all_data[idx]

        if label == "ibad":
            label = 0
        elif label == "murad":
            label = 1
        elif label == "adnan":
            label = 2

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = image.transpose((2, 0, 1))
        # Normalising the image here
        image = image / 255.0
        image = torch.from_numpy(image).float()
        return image, label


def main():
    from model import Resnet18, Resnet12
    from torchinfo import summary

    try:
        os.system("clear")
    except:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_data = torch.randn(2, 3, 224, 244)  # 2 images of dims(224, 224, 3)
    data_channels = input_data.shape[1]

    net = Resnet12(data_channels=data_channels, output_size=3)

    print(summary(net, input_data=input_data, verbose=0))

    # train_dataset = FaceDataset(osp.join("face_dataset", "train"))
    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # val_dataset = FaceDataset(osp.join("face_dataset", "val"))
    # val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # for image, label in train_dataloader:
    #     # image = image.numpy()
    #     # image = image.transpose((0, 2, 3, 1))
    #     # image = image[0]
    #     # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     # cv2.imshow("image", image)
    #     # cv2.waitKey(0)

    #     print(image.shape)
    #     print(label)

    #     break


if __name__ == "__main__":
    main()
