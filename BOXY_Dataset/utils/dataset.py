import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob
import cv2
import os
import torchvision.transforms as tfs


class ReadDataset(Dataset):
    def __init__(self, list_file_path, target_img_size=224):
        #list_file_path: the file that store the trian or val pic names
        #input_img_size means the target input size of the network
        with open(list_file_path, 'r') as f:
            self.img_file_paths = f.readlines()
        self.label_file_paths = [img_file_path.replace('png', 'txt').replace('jpg', 'txt') for img_file_path in self.img_file_paths]
        self.target_img_size = (target_img_size, target_img_size)

    def __getitem__(self, index):
        img_file_path = self.img_file_paths[index % len(self.img_file_paths)].rstrip()
        img = cv2.imread(img_file_path)

        #convert bgr to rgb
        img = img[..., ::-1]
        img = np.array(img)
        #add padding to the original picture --> h:w = 1:1
        h, w, _ = img.shape
        dimension_difference = abs(h - w)
        pad1, pad2 = dimension_difference//2, dimension_difference - dimension_difference//2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h<=w else ((0, 0), (pad1, pad2), (0, 0))
        padded_img = np.pad(img, pad, 'constant', constant_values=127.5)
        #save the padded dimensions
        padded_h, padded_w, _ = padded_img.shape
        #resize the picture to the target size
        resized_img = cv2.resize(padded_img, self.target_img_size, interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized_img)
        T = tfs.Compose([
            tfs.ColorJitter(1.5, 1.5, 1.5, .1),
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_img = T(pil_img)

        label_file_path = self.label_file_paths[index % len(self.label_file_paths)].rstrip()
        labels = None

        if os.path.exists(label_file_path):
            labels = np.loadtxt(label_file_path).reshape(-1, 8)

            # calculate the dimensions of the original picture
            x1 = (labels[:, 0] - labels[:, 2] / 2) * w
            y1 = (labels[:, 1] - labels[:, 3] / 2) * h
            x2 = (labels[:, 0] + labels[:, 2] / 2) * w
            y2 = (labels[:, 1] + labels[:, 3] / 2) * h
            p2x = (labels[:, 4]) * w
            p2y = (labels[:, 5]) * h
            p3y = (labels[:, 6]) * h
            # add the padding to the original dimensions
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            p2x += pad[1][0]
            p2y += pad[0][0]
            p3y += pad[0][0]
            # recalculate the dimensions on the new resized picture
            labels[:, 0] = ((x1 + x2) / 2) / padded_w
            labels[:, 1] = ((y1 + y2) / 2) / padded_h
            labels[:, 2] *= w / padded_w
            labels[:, 3] *= h / padded_h
            labels[:, 4] = p2x / padded_w
            labels[:, 5] = p2y / padded_h
            labels[:, 6] = p3y / padded_h
        filled_labels = np.zeros((1, 8))
        if labels is not None:
            filled_labels += labels
        filled_labels = torch.from_numpy(filled_labels)
        return img_file_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_file_paths)


class InputFolder(Dataset):
    def __init__(self, folder_path):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (224, 224)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        dimension_difference = np.abs(h - w)
        pad1, pad2 = dimension_difference // 2, dimension_difference - dimension_difference // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        padded_img = np.pad(img, pad, 'constant', constant_values=127.5)
        # save the padded dimensions
        padded_h, padded_w, _ = padded_img.shape
        # resize the picture to the target size
        resized_img = cv2.resize(padded_img, self.img_shape, interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized_img)
        T = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_img = T(pil_img)
        return img_path, input_img

    def __len__(self):
        return len(self.files)