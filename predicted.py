import os
import torch
import random
import warnings
import numpy as np
from PIL import Image
import transforms as T
import cv2
warnings.filterwarnings("ignore")
from models.mynet.CTT_CGINet import CTT_CGINet

random.seed(47)


class DataPrese:
    def __init__(self, mean=[0.406, 0.456, 0.485, 0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229, 0.225, 0.224, 0.229]):
        trans = []
        trans.extend([
            T.Normalize(mean=mean, std=std),
            T.ToTensor(),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(mean=[0.406, 0.456, 0.485, 0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229, 0.225, 0.224, 0.229]):
    return DataPrese(mean=mean, std=std)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    parser.add_argument("--ckpt_url", default="output/CDD_NEXT/best_levir_new.pth",
                        help="data root")
    parser.add_argument("--modelname", default="",
                        help="data root")
    parser.add_argument("--data_path", default="samples/CDD",
                        help="data root")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--out_path", default="./pred_best_best_cdd", help="val root")
    args = parser.parse_args()

    return args


args = parse_args()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
model = CTT_CGINet(3, 2)
checkpoint = torch.load(args.ckpt_url)
model.load_state_dict(checkpoint['model'])
model.eval()
model.to(device)

# # whu
# mean = [0.484, 0.463, 0.423, 0.484, 0.463, 0.423]
# std = [0.194, 0.188, 0.201, 0.194, 0.188, 0.201]
# LEVIR-CD
mean = [0.398,0.392,0.335,0.398,0.392,0.335]
std = [0.187,0.178,0.166,0.187,0.178,0.166]

# # old_cdd
# mean = [0.414,0.445,0.406,0.414,0.445,0.406]
# std  = [0.229,0.247,0.233,0.229,0.247,0.233]
transform = get_transform(mean,std)
# 预测并保存结果
testA_dir = os.path.join(args.data_path, "A")
testB_dir = os.path.join(args.data_path, "B")
label_dir = os.path.join(args.data_path, "label")

result_dir = args.out_path
numbers = len(os.listdir(testA_dir))
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

for root,dirs,files in os.walk(testA_dir):
    for filename in files:
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 加载图像
            image_path_A = os.path.join(root, filename)
            image_path_B = os.path.join(testB_dir, os.path.relpath(image_path_A, testA_dir))
            label_path = os.path.join(label_dir, os.path.relpath(image_path_A, testA_dir))
            label = cv2.imread(label_path,0) # 以灰度格式读取，防止通道数为3
            imageA = cv2.imread(image_path_A)[:, :, ::-1]  # 转换为RGB
            imageB = cv2.imread(image_path_B)[:, :, ::-1]
            img = np.concatenate((imageA, imageB), axis=2)
            img, label = transform(img, label)
            imageA, imageB = img[0:3].to(device), img[3:6].to(device)

        # 推理
        with torch.no_grad():
            output = model(imageA.unsqueeze(0), imageB.unsqueeze(0))
            output = torch.argmax(output, dim=1).squeeze(0)
            output = output.detach().cpu().numpy()
        output[output == 1] = 255
        output = Image.fromarray(output.astype(np.uint8))
        output.save(os.path.join(result_dir, filename))
