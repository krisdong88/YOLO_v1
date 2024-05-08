import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    """
    计算YOLO（v1）模型的损失
    """
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")  # 使用均方误差损失

        """
        S为图像分割的大小（论文中为7），
        B为每个单元格预测的边界框数量（论文中为2），
        C为类别数量（VOC数据集和论文中为20），
        """
        self.S = S
        self.B = B
        self.C = C

        # 这些是从YOLO论文中得到的，表明对于没有物体的损失(noobj)和边界框坐标的损失(coord)的权重
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # 预测的形状为 (BATCH_SIZE, S*S*(C+B*5)) 当输入
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # 计算与目标边界框的两个预测边界框的IoU
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # 从两个预测中选择IoU最高的边界框
        # bestbox将是0或1的索引，表示哪个边界框更好
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)  # 文中为 Iobj_i

        # ======================== #
        #   处理边界框坐标的损失    #
        # ======================== #

        # 对没有物体的单元格设置为0。我们只取出预测中IoU最高的一个边界框。
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30]
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]

        # 为了确保对小框的精确预测，取宽度和高度的平方根
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   处理有物体的损失    #
        # ==================== #

        # pred_box是具有最高IoU的边界框的置信度得分
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # ======================= #
        #   处理没有物体的损失    #
        # ======================= #

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ================== #
        #   处理类别的损失    #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),
            torch.flatten(exists_box * target[..., :20], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss  # 论文中的前两行
            + object_loss  # 论文中的第三行
            + self.lambda_noobj * no_object_loss  # 论文中的第四行
            + class_loss  # 论文中的第五行
        )

        return loss
