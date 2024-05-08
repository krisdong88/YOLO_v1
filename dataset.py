import torch
import os
import pandas as pd
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):
        # 初始化数据集
        self.annotations = pd.read_csv(csv_file)  # 读取CSV文件
        self.img_dir = img_dir                    # 图像文件夹路径
        self.label_dir = label_dir                # 标签文件夹路径
        self.transform = transform                # 转换函数（用于图像预处理）
        self.S = S                                # 将图像分成SxS的网格
        self.B = B                                # 每个网格的边界框数量
        self.C = C                                # 类别数量

    def __len__(self):
        # 返回数据集的大小
        return len(self.annotations)

    def __getitem__(self, index):
        # 获取单个样本
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])  # 标签文件路径
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                # 解析标签文件中的每一行
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])  # 图像文件路径
        image = Image.open(img_path)  # 加载图像
        boxes = torch.tensor(boxes)  # 将boxes转换为tensor

        if self.transform:
            # 如果存在转换函数，对图像和边界框进行预处理
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1  # 标记该网格已有边界框
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates  # 存储边界框坐标
                label_matrix[i, j, class_label] = 1  # 标记类别

        return image, label_matrix  # 返回处理后的图像和标签矩阵
