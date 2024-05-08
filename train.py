import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1  # 导入YOLOv1模型
from dataset import VOCDataset  # 导入处理Pascal VOC数据集的类
from utils import (  # 导入工具函数
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss  # 导入YOLO损失函数

# 设置随机种子以保证结果可复现
seed = 123
torch.manual_seed(seed)

# 超参数设置
LEARNING_RATE = 2e-5  # 学习率
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 设备选择
BATCH_SIZE = 16  # 批大小，原始论文中为64，根据显存调整
WEIGHT_DECAY = 0  # 权重衰减
EPOCHS = 1000  # 训练轮次
NUM_WORKERS = 2  # 数据加载时的线程数
PIN_MEMORY = True  # 是否锁页内存
LOAD_MODEL = False  # 是否加载预训练模型
LOAD_MODEL_FILE = "overfit.pth.tar"  # 加载模型的文件路径
IMG_DIR = "data/images"  # 图像文件夹路径
LABEL_DIR = "data/labels"  # 标签文件夹路径

# 定义图像变换
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes

# 定义数据预处理流程
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

# 训练函数
def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)  # 使用tqdm显示训练进度
    mean_loss = []  # 存储平均损失

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")  # 打印平均损失

# 主函数
def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset("data/100examples.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)
    test_dataset = VOCDataset("data/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True)

    for epoch in range(EPOCHS):
        print(f"{epoch}/{EPOCHS}")
        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
        print(f"Train mAP: {mean_avg_prec}")

        if mean_avg_prec > 0.9:
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            import time
            time.sleep(10)

        train_fn(train_loader, model, optimizer, loss_fn)

if __name__ == "__main__":
    main()
