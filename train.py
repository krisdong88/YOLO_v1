import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc

LEARNING_RATE = 2e-5
DEVICE = "cude" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHES = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
            return img, bboxes

transform = Compose([transforms.Resize((448,448)),transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave = True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x. to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss.fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss = loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def main():
    model = Yolov1(split_size = 7, num_boxes = 2, num_classes = 20).to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE),model, optimizer)

    train_dataset = VOCDataset(
        "8examples.csv",
        transform = transform,
        img_dir = IMG_DIR,
        lagbel_dir = LABEL_DIR
    )

    test_dataset = VOCDataset(
        "test.csv", 
        transform = transform, 
        img_dir= IMG_DIR, 
        label_dir = LABEL_DIR
    )

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle = True,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle = True,
        drop_last=True
    )

    for epoch in range(EPOCHES):
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format = "midpoint"
        )

        print(f"Train mAp: {mean_avg_prec}")
        train_fn(train_loader,model,optimizer, loss_fn)


if __name__ == "__main__":
    main()
