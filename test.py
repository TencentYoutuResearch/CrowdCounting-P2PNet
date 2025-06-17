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

    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--row', default=2, type=int,
                        help="Row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="Line number of anchor points")
    parser.add_argument('--output_dir', default='',
                        help='Path where to save output images')
    parser.add_argument('--weight_path', default='',
                        help='Path to the trained weights')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='GPU ID to use for evaluation')
    parser.add_argument('--img_dir', required=True,
                        help='Directory containing input images')
    parser.add_argument('--num_classes', default=1, type=int,
                        help="Number of distinct object classes (excluding background)")

    return parser


def main(args, debug=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    model = build_model(args)
    model.to(device)

    if args.weight_path:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    model.eval()

    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    os.makedirs(args.output_dir, exist_ok=True)

    img_dir = Path(args.img_dir)
    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))

    for img_path in img_files:
        img_raw = Image.open(img_path).convert('RGB')

        width, height = img_raw.size
        new_width = width // 128 * 128
        new_height = height // 128 * 128
        img_raw = img_raw.resize((new_width, new_height), Image.Resampling.LANCZOS)

        img = transform(img_raw)
        samples = img.unsqueeze(0).to(device)

        with torch.no_grad():  # Disable gradient computation
            outputs = model(samples)

        logits = outputs['pred_logits'][0]  # [num_queries, num_classes + 1]
        points = outputs['pred_points'][0]  # [num_queries, 2]

        # Skip class 0 (background)
        probs = torch.nn.functional.softmax(logits, -1)  # [num_queries, num_classes + 1]
        object_probs = probs[:, 1:]  # Exclude background class

        # Get max score and class index (for object classes only)
        class_scores, class_ids = torch.max(object_probs, dim=-1)

        # Filter out low-confidence predictions
        keep = class_scores > 0.5
        kept_points = points[keep].detach().cpu().numpy()
        kept_classes = class_ids[keep].detach().cpu().numpy() + 1  # shift to actual class index
        kept_scores = class_scores[keep].detach().cpu().numpy()

        count = int(keep.sum())

        # Draw predictions
        img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        for p, cls_id, score in zip(kept_points, kept_classes, kept_scores):
            x, y = int(p[0]), int(p[1])
            cv2.circle(img_to_draw, (x, y), 2, (0, 0, 255), -1)
            label = f"Class {cls_id} ({score:.2f})"
            cv2.putText(img_to_draw, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        output_path = Path(args.output_dir) / img_path.name
        cv2.imwrite(str(output_path), img_to_draw)

        print(f"[INFO] Processed {img_path.name} - Count: {count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
