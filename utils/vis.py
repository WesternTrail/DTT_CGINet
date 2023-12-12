import cv2
import numpy as np
import torch.nn.functional as F
import torch


def attention_weights_visulize(weights_dict,original_img,save_base_path='./'):
    bs,channel, height, width = original_img.shape
    alpha_att_map = F.upsample(weights_dict, (width,height))
    original_img = original_img.data.cpu().numpy()[0]
    alpha_att_map_ = cv2.applyColorMap(np.uint8(255 * alpha_att_map.data.cpu().numpy()[0][0]), cv2.COLORMAP_JET)
    fuse_heat_map = alpha_att_map_
    cv2.imwrite(save_base_path + '.jpg',fuse_heat_map)