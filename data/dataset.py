import os
from pathlib import Path
from glob import glob
from PIL import Image
import numpy as np
from transforms import train_transform, test_transform
from torch.utils.data import Dataset
os.chdir("E:/")

train_path = Path("Food101 dataset/pizza_steak_sushi/train")
test_path = Path("Food101 dataset/pizza_steak_sushi/test")

classes_name = ["pizza", "steak", "sushi"]
dictionary = {cls_name : i for i, cls_name in enumerate(classes_name)}

class Food101(Dataset):
    def __init__(self, mode):
        if mode == "train":
            self.img_list = list(train_path.glob("*/*.jpg"))
            self.transform = train_transform
        else:
            self.img_list = list(test_path.glob("*/*.jpg"))
            self.transform =test_transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image_dir = self.img_list[idx]
        image = np.array(Image.open(image_dir))
        label = dictionary[image_dir.parent.stem]
        if self.transform:
            transformed  = self.transform(image = image)
            image_transform = transformed["image"]
        return image_transform, label
    
if __name__ == "__main__":
    img_dir = Path("Food101 dataset/pizza_steak_sushi")
    data = Food101(img_dir = img_dir, transform = train_transform)
    img, label = data.__getitem__(50)
    print(img.shape)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.title(f"{label}")
    plt.show()