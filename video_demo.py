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
import imutils
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

    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')
    parser.add_argument('--video_path', default='',
                        help='path where of video')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

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
    #using Args
    """
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    """
    #Loading file directly
    checkpoint = torch.load(Path('./weights/SHTechA.pth'), map_location='cpu')
    model.load_state_dict(checkpoint['model'])


    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    '''in video'''
    fourcc = cv2.VideoWriter_fourcc(*'XVID')



    cap = cv2.VideoCapture('./fair.mp4')
    ret, frame = cap.read()
    print(frame.shape)

    '''out video'''

    scale_factor = 0.4

    width = frame.shape[1] #output size
    height = frame.shape[0] #output size
    out = cv2.VideoWriter('./demo.avi', fourcc, 30, (1280, 1280))

    while True:
        try:
            ret, frame = cap.read()


            new_width = width // 128 * 128
            new_height = height // 128 * 128

            scale_factor = 0.4
            frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            img_raw= frame.copy()
            ori_img = frame.copy()
            
        
        except:
            print("Test End")
            cap.release()
            break
        


        frame = frame.copy()

        # pre-proccessing
        img = transform(frame)
        samples = torch.Tensor(img).unsqueeze(0)
        samples = samples.to(device)

        with torch.no_grad():

            # run inference
            outputs = model(samples)
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

            outputs_points = outputs['pred_points'][0]

            threshold = 0.5
           # filter the predictions
            points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
            predict_cnt = int((outputs_scores > threshold).sum())

            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]


            outputs_points = outputs['pred_points'][0]
            print("Count: ",predict_cnt)

            # draw the predictions
            size = 2

            for p in points:
                img_to_draw = cv2.circle(img_raw , (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
            res = np.vstack((ori_img, img_to_draw))

            cv2.putText(res, "Count:" + str(predict_cnt), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # save the visualized image
            cv2.imwrite('./demo.jpg', res)
            '''write in out_video'''
            res = cv2.resize(res, (1280,1280))
            out.write(res)
            
            cv2.putText(img_to_draw, "Count:" + str(predict_cnt), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("dst",img_to_draw)
            if cv2.waitKey(1) & 0xFF == ord('q'):
               break



if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
