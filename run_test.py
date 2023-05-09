import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='./results',
                        help='path where to save') # 输出路径
    parser.add_argument('--weight_path', default='./ckpt/best_mae.pth',
                        help='path where the trained weights saved') # 权重路径

    parser.add_argument('--gpu_id', default=1, type=int, help='the gpu used for evaluation')

    return parser

def main(args, debug=False):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # set the test path here
    test_path = "/home/ipad_remote/P2PNET_ROOT/dota_split/DOTA_split_small/val/images/"
    test_img_list = os.listdir(test_path)
    # 将test_img_list按照数字排序
    test_img_list.sort(key=lambda x:int(x.split('_')[1].split('.')[0]))
    # 遍历所有图片
    for img_name in test_img_list:
        # 提取出img_name中的数字
        img_num = img_name.split('_')[1].split('.')[0]
        lines_write = img_num + ' '   
        # set your image path here
        # img_path = "./vis/P0696__1024__515___0.png" # 测试数据图片
        # load the images
        img_name = os.path.join(test_path, img_name)
        img_raw = Image.open(img_name).convert('RGB')
        # round the size
        width, height = img_raw.size
        new_width = width // 128 * 128
        new_height = height // 128 * 128
        img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
        # pre-proccessing
        img = transform(img_raw)
        samples = torch.Tensor(img).unsqueeze(0)
        samples = samples.to(device)
        # run inference
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0] # 输出的点的坐标
        threshold = 0.5 # 阈值
        # filter the predictions
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist() # 最终大于阈值的点的坐标
        predict_cnt = int((outputs_scores > threshold).sum()) # 预测的总数
        # 将predict_cnt写入lines_write
        lines_write = lines_write + str(predict_cnt) + ' '
        # 将points写入lines_write
        for point in points:
            lines_write = lines_write + str(point[0]) + ' ' + str(point[1]) + ' '
        # 将lines_write写入txt文件
        with open('results.txt', 'a') as f:
            f.write(lines_write + '\n')
        # outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0] # 计算置信度
        # outputs_points = outputs['pred_points'][0] # 输出的点的坐标
        
    # draw the predictions 注释掉这一段
    # size = 2
    # img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    # for p in points:
    #    img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    # save the visualized image
    # cv2.imwrite(os.path.join(args.output_dir, 'pred{}.jpg'.format(predict_cnt)), img_to_draw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)