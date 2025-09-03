import os
import sys
import re
import yaml
import torch
import json
import numpy as np
import argparse
from skimage.io import imread
from tqdm import tqdm
from metric_utils import Metrics
from PIL import Image

def load_image(image_path):
    """Load and normalize images."""
    image = imread(image_path).astype(np.float32) / 255.0
    return torch.tensor(image)

# def rgba2rgb(tensor_hwc):
#     """Drop the alpha channel if it exists."""
#     if tensor_hwc.size(2) == 4:  # If the tensor has 4 channels
#         return tensor_hwc[:, :, :3]  # Keep only the RGB channels
#     return tensor_hwc

def rgba2rgb(tensor_hwc):
    """Convert RGBA tensor to RGB. Set pixels where alpha == 0 to black."""
    if tensor_hwc.size(2) == 4:  # If the tensor has 4 channels
        rgb = tensor_hwc[:, :, :3]  # Keep only the RGB channels
        alpha = tensor_hwc[:, :, 3]  # Extract the alpha channel
        # Set RGB to 0 where alpha is 0
        rgb[alpha < 0.5] = 0
        return rgb
    return tensor_hwc

def get_bbox(bbox_dir, test_view, frame_id):
    js_path = os.path.join(bbox_dir, f"colmap_{frame_id}/crop_infos.json")
    with open(js_path, 'r') as file:
        data = json.load(file)
    return data[str(test_view)]

def compute_metrics(config):
    """Compute metrics based on the configuration file."""
    in_dir = config['in_dir']
    gt_pattern = re.compile(config['gt_pattern'])
    pred_pattern = re.compile(config['pred_pattern'])
    metric_list = config['metric_list']
    bbox_dir = config.get('bbox_dir', None)

    # Collect ground truth and prediction files
    gt_files = {}
    pred_files = {}
    if bbox_dir:
        gt_bboxs = {} # same as pred_bboxs
    for root, _, files in os.walk(in_dir):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), in_dir)
            match_gt = gt_pattern.match(relative_path)
            match_pred = pred_pattern.match(relative_path)
            if match_gt:
                test_view = match_gt.group(1)
                gt_files.setdefault(test_view, []).append(os.path.join(root, file))
                if bbox_dir:
                    test_view_int = int(test_view)
                    frame_id = int(match_gt.group(2))
                    bbox = get_bbox(bbox_dir, test_view_int, frame_id)
                    gt_bboxs.setdefault(test_view, []).append(bbox)
            if match_pred:
                test_view = match_pred.group(1)
                pred_files.setdefault(test_view, []).append(os.path.join(root, file))

    
    results = {test_view: {metric: [] for metric in metric_list} for test_view in gt_files.keys()}
    average_results = {metric: [] for metric in metric_list}
    for test_view in tqdm(gt_files.keys(), desc='Processing test_views'):
        if bbox_dir:
            ######### for gt ########
            # sort files and bboxes together
            combined = list(zip(gt_files[test_view], gt_bboxs[test_view]))
            # Sort the tuples based on the first list
            sorted_combined = sorted(combined, key=lambda x: x[0])
            # Unzip the sorted tuples back into separate lists
            gt_files[test_view], gt_bboxs[test_view] = zip(*sorted_combined)
            # Convert the results back to lists (since zip returns tuples)
            gt_files[test_view] = list(gt_files[test_view])
            gt_bboxs[test_view] = list(gt_bboxs[test_view])
            ######### for pred ########
            pred_files[test_view].sort()
        else:
            gt_files[test_view].sort()
            pred_files[test_view].sort()
        assert len(gt_files[test_view]) == len(pred_files[test_view]), f"Mismatch in number of files for test_view {test_view}"


        # for gt_file, pred_file in tqdm(zip(gt_files[test_view], pred_files[test_view]), total=len(gt_files[test_view]), desc=f'Processing {test_view}'):
        for i in tqdm(range(len(gt_files[test_view])), total=len(gt_files[test_view]), desc=f'Processing {test_view}'):
            gt_file = gt_files[test_view][i]
            pred_file = pred_files[test_view][i]
            gt_image = load_image(gt_file) # HWC
            pred_image = load_image(pred_file)
            # Drop the alpha channel if it exists
            gt_image = rgba2rgb(gt_image) # HW3
            pred_image = rgba2rgb(pred_image)

            if bbox_dir: # pad cropped img back to (half of) orginal size
                bbox = gt_bboxs[test_view][i]
                if True:
                  # assumes DNA imgs are down-scaled by 2
                  # assumes imgs to be evaluated (in_dir) are down-scaled original images (bbox_dir) by 2
                  full_gt = torch.zeros((bbox["ori_h"]//2, bbox["ori_w"]//2, 3), dtype=torch.float32) # 0 = black; HW3
                  full_pred = torch.zeros((bbox["ori_h"]//2, bbox["ori_w"]//2, 3), dtype=torch.float32) # 0 = black
                  assert abs(gt_image.shape[0] - bbox["bbox_xywh"][3]/2) < 1
                  assert abs(gt_image.shape[1] - bbox["bbox_xywh"][2]/2) < 1
                  full_gt[bbox["bbox_xywh"][1]//2:bbox["bbox_xywh"][1]//2+gt_image.shape[0], bbox["bbox_xywh"][0]//2:bbox["bbox_xywh"][0]//2+gt_image.shape[1], :] = gt_image
                  full_pred[bbox["bbox_xywh"][1]//2:bbox["bbox_xywh"][1]//2+gt_image.shape[0], bbox["bbox_xywh"][0]//2:bbox["bbox_xywh"][0]//2+gt_image.shape[1], :] = pred_image
                else:
                  # assumes DNA imgs are not down-scaled 
                  full_gt = torch.zeros((bbox["ori_h"], bbox["ori_w"], 3), dtype=torch.float32) # 0 = black; HW3
                  full_pred = torch.zeros((bbox["ori_h"], bbox["ori_w"], 3), dtype=torch.float32) # 0 = black
                  assert abs(gt_image.shape[0] - bbox["bbox_xywh"][3]) < 1
                  assert abs(gt_image.shape[1] - bbox["bbox_xywh"][2]) < 1
                  full_gt[bbox["bbox_xywh"][1]:bbox["bbox_xywh"][1]+gt_image.shape[0], bbox["bbox_xywh"][0]:bbox["bbox_xywh"][0]+gt_image.shape[1], :] = gt_image
                  full_pred[bbox["bbox_xywh"][1]:bbox["bbox_xywh"][1]+gt_image.shape[0], bbox["bbox_xywh"][0]:bbox["bbox_xywh"][0]+gt_image.shape[1], :] = pred_image
                gt_image = full_gt
                pred_image = full_pred

            for metric in metric_list:
                metric_fn = getattr(Metrics, metric.upper())
                result = metric_fn(gt_image, pred_image)
                results[test_view][metric].append(result)
                average_results[metric].append(result)

    # Compute averages for each test_view
    for test_view in results.keys():
        for metric in metric_list:
            results[test_view][metric] = np.mean(results[test_view][metric])

    # Compute overall averages
    for metric in metric_list:
        average_results[metric] = np.mean(average_results[metric])

    results['average'] = average_results

    # Save results to a JSON file
    this_files_dir = os.path.dirname(os.path.realpath(__file__))
    res_dir = os.path.join(this_files_dir, "results")
    output_path = os.path.join(res_dir, config['output_file'])
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as file:
        json.dump(results, file, indent=4)

    # Print results
    for test_view, metrics in results.items():
        print(f"test_view: {test_view}")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluation metrics calculation')
    parser.add_argument('--config', type=str, default=None, help='Path to the configuration YAML file.')
    parser.add_argument('--in_dir', type=str,
                       default=None,
                       help='Input directory containing the images')
    parser.add_argument('--gt_pattern', type=str,
                       default=r'test/it30000/v(\d{2})/gt_f\d{4}.png',
                       help='Regex pattern for ground truth images')
    parser.add_argument('--pred_pattern', type=str,
                       default=r'test/it30000/v(\d{2})/pred_f\d{4}.png',
                       help='Regex pattern for predicted images')
    parser.add_argument('--metric_list', type=str,
                       nargs='+',
                       default=['ssim', 'psnr', 'lpips_alex', 'lpips_vgg'],
                       help='List of metrics to compute')
    parser.add_argument('--output_file', type=str,
                       default=None,
                       help='Output JSON file path for metrics results')
    args = parser.parse_args()
    
    # If config exists, use it as base, otherwise use args
    if args.config is not None and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Convert args to dictionary
        config = vars(args)
        # Remove config from dictionary as it's not needed for compute_metrics
        config.pop('config', None)

    
    compute_metrics(config)

if __name__ == '__main__':
    main()
