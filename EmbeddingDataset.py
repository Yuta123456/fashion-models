import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms

class EmbeddingDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        # labels.csv => filepath, label の形式になっている
        # ここをfilepath, textにすればいい？
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform =  transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)

        # lstmに入れる形にする
        caption = self.img_labels.iloc[idx, 1]
        
        sample = {"image": image, "caption": caption}
        return sample
    