from torch.utils.data import Dataset
from torchvision.transforms import *
import os
import random
import torch
import numpy as np
from PIL import Image
import csv
from matplotlib import pyplot as plt

class EMPS_TEST(Dataset):
    def __init__(self, root_dir: str, transforms: object = None):
        super().__init__()

        self.root_dir = root_dir
        self.transforms_mask = transforms
        self.transforms_img = transforms

        self.mask_dir = os.path.join(root_dir, "segmaps")
        self.img_dir = os.path.join(root_dir, "images")

        # self.file_names = os.listdir(self.mask_dir)
        self.file_names = self._getTestFilename()
        self.mask_paths = [os.path.join(self.mask_dir, mask_name) for mask_name in self.file_names]
        self.img_paths = [os.path.join(self.img_dir, img_name) for img_name in self.file_names]

    def _getTestFilename(self):
        testCsv_path = os.path.join(self.root_dir, "test.csv")
        test_file_names = []
        with open(testCsv_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                test_file_names.append(row[0]+'.png')
        return test_file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, ix: int):
        seed = np.random.randint(2022)
        random.seed(seed)
        torch.manual_seed(seed)

        mask_path, img_path = self.mask_paths[ix], self.img_paths[ix]

        mask, img = Image.open(mask_path), Image.open(img_path)
        # 确保图像是 'RGB' 模式且掩码是 'L' 模式
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if mask.mode != 'L':
            mask = mask.convert('L')

        if self.transforms_img is not None:
            img = self.transforms_img(img)  # img:(3,224,224)

        random.seed(seed)
        torch.manual_seed(seed)

        if self.transforms_mask is not None:
            mask = self.transforms_mask(mask)   # mask:(1,224,224)

        mask, img = mask.long(), img[[0]].float()   # mask:(1,224,224)   img:(1,224,224)

        mask[mask > 0] = 1
        img = img / 255

        return img, mask

transforms = Compose([
    # Resize((256,256)),
    Resize((224, 224)),
    RandomRotation(45),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    GaussianBlur(5),
    PILToTensor(),
])

if __name__ == '__main__':
    dataset = EMPS_TEST(root_dir="E:/Projects/_Machine_Learning/DataSet/emps", transforms=transforms)
    img, mask = dataset[10]
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(img.permute((1, 2, 0)), cmap='gray')
    ax[1].imshow(mask.permute((1, 2, 0)), cmap='gray')
    plt.show()