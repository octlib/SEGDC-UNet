from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch
import torchvision
from torch import nn, optim
from SEGDC import SEGDC_Unet
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import csv
import os
from datetime import datetime
from sklearn.metrics import jaccard_score, recall_score
from dataset_EMPS_test import EMPS_TEST, transforms


def check_accuracy(loader, model, device='cuda'):
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

            # Calculate recall
            recall += recall_score(y.cpu().numpy().flatten(), preds.cpu().numpy().flatten())

            # Calculate IoU
            iou += jaccard_score(y.cpu().numpy().flatten(), preds.cpu().numpy().flatten())

    if pixels == 0:  # Avoid division by zero
        return 0, 0, 0, 0

    test_accuracy = correct / pixels
    dice_score /= len(loader)
    recall /= len(loader)
    iou /= len(loader)

    print(f"Accuracy: {100 * test_accuracy:.2f}")
    print(f"Dice score: {dice_score:.4f}")
    print(f"Recall: {recall:.2f}")
    print(f"IoU: {iou:.2f}")

    model.train()
    return test_accuracy, dice_score, recall, iou

def check_img_predictions(loader, model, path="./results", device='cuda'):
    model.eval()
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = model(x)
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{path}/pred_res_{i}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{path}/y_res_{i}.png")
    model.train()


###############################################################################################
#Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4

# 数据集
dataset_dir = 'E:/Projects/_Machine_Learning/DataSet/emps'
dataset = EMPS_TEST(root_dir=dataset_dir, transforms=transforms)
dataloader_test = DataLoader(dataset, batch_size=batch_size)

# 模型
model = SEGDC_Unet(in_channels=1, out_channels=1).to(device) #,features=[16,32,64,128]).to(device)
model_state_dict = torch.load('best_model.pth.tar')['model_state_dict']
model.load_state_dict(model_state_dict)

# 损失函数
loss_fn = nn.BCEWithLogitsLoss()

# 创建带时间戳的子文件夹
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
test_results_dir = os.path.join("results", "test", timestamp)
os.makedirs(test_results_dir, exist_ok=True)

# 在测试开始时创建CSV文件并写入列标题
csv_filename = os.path.join(test_results_dir, "test_metrics.csv")
with open(csv_filename, mode='w', newline='') as csv_file:
    fieldnames = ['Val_Accuracy', 'Val_Dice_score', 'Val_Recall', 'Val_IoU']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

if __name__ == "__main__":
    # 测试结果
    test_accuracy, test_dice_score, test_recall, test_iou = check_accuracy(dataloader_test, model, device)

    # 保存指标到CSV文件
    with open(csv_filename, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({'Val_Accuracy': test_accuracy,
                         'Val_Dice_score': test_dice_score,
                         'Val_Recall': test_recall,
                         'Val_IoU': test_iou})

    # 查看预测结果图（预测图、标注图、原始图）
    i = np.random.randint(1)
    for x, target in dataloader_test:
        pred = model(x[[i]].to(device)).cpu().squeeze().detach()
        pred = torch.where(pred > 0.5, 1, 0).squeeze()
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 16))
        ax[0].imshow(pred, cmap='gray')
        ax[1].imshow(target[[i]].squeeze())
        ax[2].imshow(x[[i]].squeeze())
        plt.show()

        # +break：只显示一幅图，-break：所有测试集图全显示
        break
