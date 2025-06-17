import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io

class SHHA(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
        self.root_path = data_root
        self.train_lists = "train.list"
        self.eval_list = "test.list"
        # there may exist multiple list files
        self.img_list_file = self.train_lists.split(',')
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')

        self.img_map = {}
        self.img_list = []
        # loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            with open(os.path.join(self.root_path, train_list)) as fin:
                for line in fin:
                    if len(line) < 2: 
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                                    os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))
        # number of samples
        self.nSamples = len(self.img_list)
        
        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        # load image and ground truth
        img, points_with_labels = load_data((img_path, gt_path), self.train)
        # applu augumentation
        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            # data augmentation -> random scale
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            # scale the image and points
            if scale * min_size > 128:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                points_with_labels[:, :2] *= scale # Scale only x, y coordinates
        # random crop augumentaiton
        if self.train and self.patch:
            # random_crop now returns result_img, result_points_with_labels
            img, points_with_labels_list = random_crop(img, points_with_labels)
            # Process each item in the list (each patch)
            processed_points_with_labels = []
            for item in points_with_labels_list:
                processed_points_with_labels.append(torch.Tensor(item))
            points_with_labels = processed_points_with_labels
        else: # If not patching, ensure points_with_labels is a list for consistency
            points_with_labels = [torch.Tensor(points_with_labels)]

        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(points_with_labels):
                points_with_labels[i][:, 0] = 128 - points_with_labels[i][:, 0]

        img = torch.Tensor(img)
        # pack up related infos
        target = [{} for _ in range(len(points_with_labels))]
        for i, current_points_with_labels in enumerate(points_with_labels):
            # Ensure current_points_with_labels is a tensor
            if not isinstance(current_points_with_labels, torch.Tensor):
                current_points_with_labels = torch.Tensor(current_points_with_labels)

            target[i]['point'] = current_points_with_labels[:, :2] # x, y coordinates
            image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            # Use class_id from points_with_labels, convert to int, and add 1 (1-indexed)
            labels = current_points_with_labels[:, 2].long() + 1
            target[i]['labels'] = labels

        return img, target


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    # load the images
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # load ground truth points
    points_with_labels = []
    with open(gt_path) as f_label:
        for line in f_label:
            parts = line.strip().split(' ')
            x = float(parts[0])
            y = float(parts[1])
            class_id = int(parts[2]) # Assuming class_id is an integer
            points_with_labels.append([x, y, class_id])

    return img, np.array(points_with_labels)

# random crop augumentation
def random_crop(img, points_with_labels, num_patch=4):
    half_h = 128
    half_w = 128
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_points_with_labels = []
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        # copy the cropped points and labels
        idx = (points_with_labels[:, 0] >= start_w) & (points_with_labels[:, 0] <= end_w) & \
              (points_with_labels[:, 1] >= start_h) & (points_with_labels[:, 1] <= end_h)
        # shift the corrdinates and keep labels
        record_points_with_labels = points_with_labels[idx]
        record_points_with_labels[:, 0] -= start_w
        record_points_with_labels[:, 1] -= start_h

        result_points_with_labels.append(record_points_with_labels)

    return result_img, result_points_with_labels