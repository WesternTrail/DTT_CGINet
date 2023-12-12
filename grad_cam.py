import os
import warnings

import cv2
import torch
import requests
import torchvision
import torch.functional as F
import numpy as np
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

from models.CTT_CGINet import CTT_CGINet

image_file1 = "samples/LEVIR-CD/B/test_2_1.png"
image_file2 = "samples/LEVIR-CD/A/test_2_1.png"
target_file = "samples/LEVIR-CD/label/test_2_1.png"
image1 = Image.open(image_file1)
image2 = Image.open(image_file2)
target = cv2.imread(target_file, 0)  # 以灰度格式读取，防止通道数为3

rgb_img1 = np.float32(image1) / 255
rgb_img2 = np.float32(image2) / 255
target = np.float32(image2)
input_tensor1 = preprocess_image(rgb_img1,
                                mean=[0.398, 0.392, 0.335],
                                std=[0.187, 0.178, 0.166])
input_tensor2 = preprocess_image(rgb_img2,
                                 mean=[0.398, 0.392, 0.335],
                                 std=[0.187, 0.178, 0.166])

model = CTT_CGINet(3, 2)
checkpoint = torch.load("best_levir_new.pth")
model.load_state_dict(checkpoint['model'])
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device=device)
input_tensor1 = input_tensor1.to(device)
input_tensor2 = input_tensor2.to(device)


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x1,x2):
        return self.model(x1,x2)

def hook_fn(module, input1,input2,output):
    # 在这里对特征图进行操作，例如输出特征图的形状
    print(f"Feature map shape:")

output = model(input_tensor1,input_tensor2)

normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
sem_classes = [
    'background',
    'crack',
]
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

crack_category = sem_class_to_idx["background"]
crack_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
crack_mask_uint8 = 255 * np.uint8(crack_mask == crack_category)
crack_mask_float = np.float32(crack_mask == crack_category)

both_images = np.hstack((image1, np.repeat(crack_mask_uint8[:, :, None], 3, axis=-1)))
img = Image.fromarray(both_images)
# img.show()

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category - 1, :, :] * self.mask).sum()

def reshape_transform(in_tensor):
    result = in_tensor.reshape(in_tensor.size(0),
                               int(np.sqrt(in_tensor.size(1))), int(np.sqrt(in_tensor.size(1))), in_tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result

target_layers = [model.classifier[1]]
targets = [SemanticSegmentationTarget(crack_category, crack_mask_float)]
with GradCAM(model=model,
             target_layers=target_layers,
             use_cuda=torch.cuda.is_available(),
             # reshape_transform=reshape_transform # 该部分是针对Vit系列模型的相关变换参数,cnn模型可不设置.
             ) as cam:
    grayscale_cam = cam(input_tensor1,input_tensor2,targets=targets)[0,:]
    cam_image = show_cam_on_image(np.float32(image1) / 255, grayscale_cam, use_rgb=True)

cam_img = Image.fromarray(cam_image)
cam_img.show()
