import glob
import os
import cv2
import numpy as np
from torch.utils.data import Dataset

res_shape = (256, 256)

class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.img_path_A = glob.glob(os.path.join(data_path, 'A', '*.[pjP][pnN][gG]'))
        self.img_path_B = glob.glob(os.path.join(data_path, 'B', '*.[pjP][pnN][gG]'))
        self.mask_path = glob.glob(os.path.join(data_path, 'label', '*.[pjP][pnN][gG]'))
        self.transforms = transform

    def get_image_paths(self, root_dir):
        image_paths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                # if file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
        return image_paths

    def __getitem__(self, index):
        images1 = cv2.imread(self.img_path_A[index])[:, :, ::-1]  # BGR to RGB
        images2 = cv2.imread(self.img_path_B[index])[:, :, ::-1]  # BGR to RGB
        label = cv2.imread(self.mask_path[index], 0)
        img = np.concatenate((images1, images2), axis=2)
        if self.transforms:
            [img, labels] = self.transforms(img, label)
        return img[0:3], img[3:], labels

    def __len__(self):
        return len(self.img_path_A)


def Mydataset_collate(batch):
    images1 = []
    images2 = []
    masks = []
    for img1, img2, mask in batch:
        images1.append(img1)
        images2.append(img2)
        masks.append(mask)
    images1 = np.array(images1)
    images2 = np.array(images2)
    masks = np.array(masks)
    return images1, images2, masks
