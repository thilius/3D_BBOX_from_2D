import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob
import cv2
import random
import torchvision.transforms as tfs


def random_patch_picture(img):
    patch_xc = random.randint(20, 204)
    patch_yc = random.randint(20, 204)
    patch_x1 = max(patch_xc - 30, 0)
    patch_y1 = max(patch_yc - 30, 0)
    patch_x2 = min(patch_xc + 30, 224)
    patch_y2 = min(patch_yc + 30, 224)
    img[patch_y1: patch_y2, patch_x1: patch_x2] = [random.randint(0, 255), random.randint(0, 255),
                                                   random.randint(0, 255)]
    return img


class ReadDataset(Dataset):
    def __init__(self, list_file_path, target_img_size=224):
        # list_file_path: the file that store the trian or val pic names
        # input_img_size means the target input size of the network
        with open(list_file_path, 'r') as f:
            self.img_file_paths = f.readlines()
        self.label_file_paths = [img_file_path.replace('png', 'txt').replace('jpg', 'txt') for img_file_path in
                                 self.img_file_paths]
        self.target_img_size = (target_img_size, target_img_size)

    def __getitem__(self, index):
        img_file_path = self.img_file_paths[index % len(self.img_file_paths)].rstrip()
        img = cv2.imread(img_file_path)

        # convert bgr to rgb
        img = img[..., ::-1]
        img = np.array(img)
        # add padding to the original picture --> h:w = 1:1
        h, w, _ = img.shape
        dimension_difference = abs(h - w)
        pad1, pad2 = dimension_difference // 2, dimension_difference - dimension_difference // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        padded_img = np.pad(img, pad, 'constant', constant_values=127.5)
        # save the padded dimensions
        padded_h, padded_w, _ = padded_img.shape
        # resize the picture to the target size
        resized_img = cv2.resize(padded_img, self.target_img_size, interpolation=cv2.INTER_AREA)

        patched_img = random_patch_picture(resized_img)

        pil_img = Image.fromarray(patched_img)
        T = tfs.Compose([
            tfs.ColorJitter(1.5, 1.5, 1.5, .1),
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_img = T(pil_img)

        label_file_path = self.label_file_paths[index % len(self.label_file_paths)].rstrip()
        labels = None

        labels = np.loadtxt(label_file_path).reshape(-1, 10)

        # calculate the dimensions of the original picture
        labels[:, 0] *= w
        labels[:, 1] *= h
        labels[:, 5] *= w
        labels[:, 6] *= h

        # add the padding to the original dimensions
        labels[:, 0] += pad[1][0]
        labels[:, 1] += pad[0][0]
        labels[:, 5] += pad[1][0]
        labels[:, 6] += pad[0][0]

        # recalculate the dimensions on the new resized picture
        labels[:, 0] /= padded_w
        labels[:, 1] /= padded_h
        labels[:, 3] *= w / padded_w
        labels[:, 4] *= h / padded_h
        labels[:, 5] /= padded_w
        labels[:, 6] /= padded_h
        labels[:, 8] *= w / padded_w
        labels[:, 9] *= h / padded_h

        filled_labels = torch.from_numpy(labels)
        return img_file_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_file_paths)

