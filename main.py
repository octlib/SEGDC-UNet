import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
from SEGDC import SEGDC_Unet
from torchvision.transforms import Compose, Resize, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, GaussianBlur, PILToTensor
from PIL import Image
import random
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import jaccard_score, recall_score
from datetime import datetime
import csv

class EMPS(Dataset):
    def __init__(self, root_dir: str, transforms: object = None):
        super().__init__()
        self.root_dir = root_dir
        self.transforms_mask = transforms
        self.transforms_img = transforms
        self.mask_dir = os.path.join(root_dir, "segmaps")
        self.img_dir = os.path.join(root_dir, "images")
        self.file_names = os.listdir(self.mask_dir)
        self.mask_paths = [os.path.join(self.mask_dir, mask_name) for mask_name in self.file_names]
        self.img_paths = [os.path.join(self.img_dir, img_name) for img_name in self.file_names]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, ix: int):
        seed = np.random.randint(2022)
        random.seed(seed)
        torch.manual_seed(seed)

        mask_path, img_path = self.mask_paths[ix], self.img_paths[ix]
        mask, img = Image.open(mask_path), Image.open(img_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')
        if mask.mode != 'L':
            mask = mask.convert('L')

        if self.transforms_img is not None:
            img = self.transforms_img(img)

        random.seed(seed)
        torch.manual_seed(seed)

        if self.transforms_mask is not None:
            mask = self.transforms_mask(mask)

        mask, img = mask.long(), img[[0]].float()
        mask[mask > 0] = 1
        img = img / 255
        return img, mask

augment_transforms = Compose([
    Resize((224, 224)),
    RandomRotation(45),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    GaussianBlur(5),
    PILToTensor(),
])

dataset = EMPS(root_dir='', transforms=augment_transforms)

# 设置随机种子以确保每次运行时数据集划分相同
seed = 1234
torch.manual_seed(seed)
random.seed(seed)
# Split the dataset into train, validation, and test sets
dataset_length = len(dataset)
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
train_length = int(train_ratio * dataset_length)
val_length = int(val_ratio * dataset_length)
test_length = dataset_length - train_length - val_length
dataset_train, dataset_val, dataset_test = random_split(dataset, [train_length, val_length, test_length])
dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=8, shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=False)


LR = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8
epochs = 50
LOAD_MODEL = False
def train(loader, model, optimizer, loss_fn, scaler):
    model.train()
    for batch, (imgs, masks) in enumerate(tqdm(loader)):
        imgs = imgs.to(device)
        masks = masks.float().to(device)

        # forward path
        preds = model(imgs)
        loss = loss_fn(preds, masks)

        # backward path
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

model = SEGDC_Unet(in_channels=1).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("-> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("-> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device="cuda"):
    model.eval()
    correct, pixels = 0, 0
    dice_score = 0.0
    recall = 0.0
    iou = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            preds = (preds > 0.5).float()
            correct += (preds == y).sum().item()
            pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum().item()) / ((preds + y).sum().item() + 1e-8)

            recall += recall_score(y.cpu().numpy().flatten(), preds.cpu().numpy().flatten())
            iou += jaccard_score(y.cpu().numpy().flatten(), preds.cpu().numpy().flatten())

    if pixels == 0:  # Avoid division by zero
        return 0, 0, 0, 0

    val_accuracy = correct / pixels
    dice_score /= len(loader)
    recall /= len(loader)
    iou /= len(loader)

    print(f"Accuracy: {100 * val_accuracy:.2f}")
    print(f"Dice score: {dice_score:.4f}")
    print(f"Recall: {recall:.2f}")
    print(f"IoU: {iou:.2f}")

    model.train()
    return val_accuracy, dice_score, recall, iou



def save_model(model: nn.Module, optimizer: optim.Optimizer, filename: str) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

if LOAD_MODEL:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
scaler = torch.cuda.amp.GradScaler()

# 创建带时间戳的子文件夹
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
train_results_dir = os.path.join("results", "train", timestamp)
os.makedirs(train_results_dir, exist_ok=True)

# 在训练开始时创建CSV文件并写入列标题
csv_filename = os.path.join(train_results_dir, "training_metrics.csv")
with open(csv_filename, mode='w', newline='') as csv_file:
    fieldnames = ['Epoch', 'Val_Accuracy', 'Val_Dice_score', 'Val_Recall', 'Val_IoU', 'Test_Accuracy', 'Test_Dice_score', 'Test_Recall', 'Test_IoU']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

best_dice_score = 0  # 初始化最佳Dice分数
best_model_path = os.path.join(train_results_dir, "best_model.pth.tar")


if __name__ == "__main__":
    for epoch in range(epochs):
        train(dataloader_train, model, optimizer, loss_fn, scaler)
        val_accuracy, val_dice_score, val_recall, val_iou = check_accuracy(dataloader_val, model, device)
        test_accuracy, test_dice_score, test_recall, test_iou = check_accuracy(dataloader_test, model, device)

        # 保存指标到CSV文件
        with open(csv_filename, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({'Epoch': epoch,
                             'Val_Accuracy': val_accuracy,
                             'Val_Dice_score': val_dice_score,
                             'Val_Recall': val_recall,
                             'Val_IoU': val_iou,
                             'Test_Accuracy': test_accuracy,
                             'Test_Dice_score': test_dice_score,
                             'Test_Recall': test_recall,
                             'Test_IoU': test_iou})

