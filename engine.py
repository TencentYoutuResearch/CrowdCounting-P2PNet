# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from util.misc import NestedTensor
import numpy as np
import time
import torchvision.transforms as standard_transforms
import cv2

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def vis(samples, targets, pred, vis_dir, des=None):
    '''
    samples -> tensor: [batch, 3, H, W]
    targets -> list of dict: [{'points':[], 'image_id': str}]
    pred -> list: [num_preds, 2]
    '''
    gts = [t['point'].tolist() for t in targets]

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])
    # draw one by one
    for idx in range(samples.shape[0]):
        sample = restore_transform(samples[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_gt = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        sample_pred = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        max_len = np.max(sample_gt.shape)

        size = 2
        # draw gt
        for t in gts[idx]:
            sample_gt = cv2.circle(sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)
        # draw predictions
        for p in pred[idx]:
            sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        name = targets[idx]['image_id']
        # save the visualized images
        if des is not None:
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_gt.jpg'.format(int(name), 
                                                des, len(gts[idx]), len(pred[idx]))), sample_gt)
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_pred.jpg'.format(int(name), 
                                                des, len(gts[idx]), len(pred[idx]))), sample_pred)
        else:
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_gt.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_gt)
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_pred.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_pred)

# the training routine
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # iterate all training samples
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # forward
        outputs = model(samples)
        # calc the losses
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce all losses
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        # backward
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # update logger
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# the inference routine
@torch.no_grad()
def evaluate_crowd_no_overlap(model, data_loader, device, vis_dir=None):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # Assuming class_error is not the primary metric for multi-class crowd counting here.
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    # Determine N (number of object classes) from the model output structure
    # This needs to be done once, can be outside the loop if model structure is fixed
    # Or fetched from model config if available. For now, infer from first output.
    _model_output_checked = False
    N = 0

    # Initialize lists for overall and per-class metrics
    overall_maes_list = []
    overall_mses_list = []
    # Deferred initialization for per_class lists after N is known

    for i, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        # Targets are already expected to be on device by this point based on train_one_epoch
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        if not _model_output_checked:
            if 'pred_logits' not in outputs or outputs['pred_logits'].ndim < 3:
                raise ValueError("pred_logits not found in model output or has unexpected shape.")
            N = outputs['pred_logits'].shape[-1] - 1 # N object classes + 1 background
            if N <= 0:
                raise ValueError(f"Inferred N={N} classes. Must be > 0.")
            print(f"Inferred N = {N} object classes from model output.")
            # Now initialize per-class metric lists
            maes_per_class_list = [[] for _ in range(N + 1)] # Index 0 unused
            mses_per_class_list = [[] for _ in range(N + 1)] # Index 0 unused
            _model_output_checked = True

        # Assuming batch size is 1 for evaluation, as in original code
        # Process Model Outputs
        probs = torch.nn.functional.softmax(outputs['pred_logits'][0], dim=-1) # Shape: (num_queries, N+1)
        pred_class_labels = torch.argmax(probs, dim=1) # Shape: (num_queries,) values 0 to N
        pred_scores = probs.max(dim=1)[0] # Shape: (num_queries,)
        outputs_points = outputs['pred_points'][0] # Shape: (num_queries, 2)

        # Filter Predictions
        threshold = 0.5 # As used in original code
        valid_scores_mask = pred_scores > threshold
        valid_class_mask = pred_class_labels > 0 # Ignore background predictions (class 0)
        final_mask = valid_scores_mask & valid_class_mask

        filtered_pred_class_labels = pred_class_labels[final_mask] # 1-indexed effectively due to valid_class_mask
        filtered_points = outputs_points[final_mask]

        # Calculate Per-Class Predicted Counts
        predict_counts_per_class = torch.zeros(N + 1, device=device, dtype=torch.float) # Idx 0 for bg (unused), 1 to N
        for label_idx in filtered_pred_class_labels:
            # pred_class_labels are 0 to N. Since we filtered >0, these are 1 to N.
            predict_counts_per_class[label_idx] += 1

        # Calculate Per-Class Ground Truth Counts
        # gt_labels are 1-indexed (1 to N) as per problem description
        gt_labels = targets[0]['labels'] # Assuming batch_size 1, labels are on device
        gt_counts_per_class = torch.zeros(N + 1, device=device, dtype=torch.float) # Idx 0 for bg (unused), 1 to N
        for label_idx in gt_labels:
            if 1 <= label_idx <= N: # Ensure labels are within expected range
                gt_counts_per_class[label_idx] += 1
            # else: print(f"Warning: GT label {label_idx} out of range 1-{N}")


        # Calculate MAE and MSE Metrics for the current sample
        for c in range(1, N + 1): # For each object class (1 to N)
            mae_c = torch.abs(predict_counts_per_class[c] - gt_counts_per_class[c])
            mse_c = (predict_counts_per_class[c] - gt_counts_per_class[c])**2
            maes_per_class_list[c].append(mae_c.item())
            mses_per_class_list[c].append(mse_c.item())

        total_predicted_count = predict_counts_per_class[1:].sum() # Sum over object classes 1 to N
        total_gt_count = gt_counts_per_class[1:].sum() # Sum over object classes 1 to N

        overall_mae = torch.abs(total_predicted_count - total_gt_count)
        overall_mse = (total_predicted_count - total_gt_count)**2
        overall_maes_list.append(overall_mae.item())
        overall_mses_list.append(overall_mse.item())

        # if specified, save the visualized images
        # Update vis call: pass filtered_points.
        # For multi-class vis, vis function would need enhancement. For now, pass filtered points.
        if vis_dir is not None:
            # vis expects a list of lists of points if batch > 1, here batch is 1
            vis(samples, targets, [filtered_points.detach().cpu().numpy().tolist()], vis_dir, des=f"sample_{i}")

    # Aggregate and Return Metrics (after the loop)
    results = {}
    if not _model_output_checked : # Handle case where data_loader was empty
        print("Warning: Data loader was empty, cannot determine N or compute metrics.")
        # Return empty or default results
        results['overall_mae'] = 0
        results['overall_mse'] = 0
        return results

    for c in range(1, N + 1):
        results[f'mae_class_{c}'] = np.mean(maes_per_class_list[c]) if maes_per_class_list[c] else 0
        results[f'mse_class_{c}'] = np.sqrt(np.mean(mses_per_class_list[c])) if mses_per_class_list[c] else 0

    results['overall_mae'] = np.mean(overall_maes_list) if overall_maes_list else 0
    # MSE in this context is usually RMSE for crowd counting (sqrt of mean of squared errors)
    results['overall_mse'] = np.sqrt(np.mean(overall_mses_list)) if overall_mses_list else 0

    # Log overall MAE/MSE using metric_logger if desired, though train.py doesn't use it for eval log
    # metric_logger.update(overall_mae=results['overall_mae'])
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger) # Not standard for this eval function

    return results