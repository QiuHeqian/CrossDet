import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import mmcv
#from mmdet.apis import init_dist
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.core import bbox_overlaps


def parse_args():
    parser = argparse.ArgumentParser(description='visualize iou distribution between stages')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('roi_file', help='checkpoint file')
    args = parser.parse_args()
    return args


def get_roi_max_overlaps(args, iou_thrs=0.3):
    cfg = mmcv.Config.fromfile(args.config)

    cfg.data.test.test_mode = True
    cfg.data.test.pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True, skip_img_without_anno=False),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='ToTensor', keys=['gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
             meta_keys=('filename', 'ori_shape', 'img_shape'))
    ]
    cfg.data.test.get_ann_at_test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    dataset = data_loader.dataset

    dataset_stage_rois = mmcv.load(args.roi_file)
    outputs = []

    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        gt_bboxes = data['gt_bboxes'][0]
        num_gt = gt_bboxes.size(0)
        img_rois = dataset_stage_rois[i]
        # del img_rois['stage2']
        img_max_overlaps = {stage_name: [] for stage_name in img_rois}
        if torch.any(gt_bboxes != 0):
            # 1. use the first stage rois (proposals from RPN) to get valid data
            init_proposals = torch.from_numpy(img_rois['stage0'])
            init_overlaps = bbox_overlaps(gt_bboxes, init_proposals)
            max_overlaps, argmax_overlaps = init_overlaps.max(dim=0)
            is_valid = (max_overlaps >= iou_thrs)
            # del init_proposals, init_overlaps, max_overlaps, argmax_overlaps
            if torch.any(is_valid):
                for stage_name in img_rois:
                    stage_rois = torch.from_numpy(img_rois[stage_name])
                    good_stage_rois = stage_rois[is_valid, :]
                    good_roi_overlaps = bbox_overlaps(gt_bboxes, good_stage_rois)
                    good_roi_max_overlaps, good_roi_argmax_overlaps = good_roi_overlaps.max(dim=0)
                    # calculate mean iou of rois and dets
                    n_valid_overlaps_roi = torch.sum((good_roi_overlaps >= iou_thrs).float(), dim=0)
                    n_valid_overlaps_roi[n_valid_overlaps_roi == 0] = 1e4
                    good_roi_overlaps[good_roi_overlaps < iou_thrs] = 0.0
                    mean_overlaps_roi = torch.sum(good_roi_overlaps, dim=0) / n_valid_overlaps_roi
                    # mean_overlaps_roi = mean_overlaps_roi[mean_overlaps_roi > iou_thrs]
                    # num_gts = gt_bboxes.shape[0]
                    # is_valid_rois = (mean_overlaps_roi >= 1e-4 * num_gts)
                    # mean_overlaps_roi = mean_overlaps_roi[is_valid_rois]
                    # mean_overlaps_roi = mean_overlaps_roi.numpy()
                    # img_max_overlaps[stage_name] = mean_overlaps_roi
                    mean_overlaps_roi = mean_overlaps_roi[mean_overlaps_roi >= iou_thrs]
                    img_max_overlaps[stage_name] = mean_overlaps_roi.numpy()

                # remove outliers: dets with very small iou
                # outliers = np.where(img_max_overlaps['stage1'] < 1e-4 * num_gt)
                # for stage_name in img_rois:
                #     img_max_overlaps[stage_name] = np.delete(img_max_overlaps[stage_name], outliers)

        outputs.append(img_max_overlaps)

        batch_size = data['img'].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    return outputs


def main():
    args = parse_args()
    iou_thrs = 0.5
    # num_bins = int((1.0-iou_thrs)/0.1)
    num_bins = 30

    max_overlaps = get_roi_max_overlaps(args, iou_thrs)
    all_stage_ious = dict()
    for stage_name in max_overlaps[0]:
        all_stage_ious[stage_name] = \
            np.concatenate([img_ious[stage_name] for img_ious in max_overlaps])
    plt.figure(figsize=(8, 5))
    n, bins, patches = plt.hist(list(all_stage_ious.values()), bins=num_bins,
                                color=['orange', 'royalblue', 'limegreen', 'plum'])
    plt.legend(['RPN proposal', '1st step', '2nd step', '3rd step'])
    plt.xlabel('IoU with gt boxes')
    plt.ylabel('# amount of boxes')
    plt.title('Box IoU distribution in 3-step recurrent detection')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # plt.axvline(0.5, color='k', linewidth=1.5, linestyle='--')
    # plt.axvline(0.7, color='r', linewidth=1.5, linestyle='--')
    plt.savefig('iou_hist_original.png', bbox_inches='tight', dpi=300)
    plt.show()

    if 0:
        iou_bin_edges, iou_step = np.linspace(iou_thrs, 1., num_bins, endpoint=True, retstep=True)
        iou_stats = np.zeros((num_bins,), dtype=np.int64)
        iou_stats = np.repeat(iou_stats[None,:], 3, axis=0)

        num_images = len(max_overlaps)
        prog_bar = mmcv.ProgressBar(num_images)
        for i in range(num_images):
            img_max_overlaps = max_overlaps[i]
            for k, stage_name in enumerate(img_max_overlaps):
                iou_labels = np.digitize(img_max_overlaps[stage_name], iou_bin_edges, right=False)
                bin_labels, label_counts = np.unique(iou_labels, return_counts=True)
                valid_bin_labels = bin_labels[bin_labels > 0]
                valid_label_counts = label_counts[bin_labels > 0]
                iou_stats[k][valid_bin_labels - 1] += valid_label_counts
            prog_bar.update()

        stage_iou_stats = [(k, v.squeeze()) for (k, v) in
                           zip(max_overlaps[0].keys(), np.vsplit(iou_stats, 3))]
        stage_iou_stats = dict(stage_iou_stats)
        for stage_name in stage_iou_stats:
            plt.figure(figsize=(5, 4))
            s = 0.05
            new_xticklabels = np.arange(iou_thrs, 1.+s, s)
            new_xticklabels = np.around(new_xticklabels, decimals=2)
            num_xticks = new_xticklabels.shape[0]
            new_xticks = np.arange(num_xticks)
            new_xticks = (new_xticks * (num_bins-1)) / (num_xticks-1)
            new_xticks = np.around(new_xticks, decimals=1)
            plt.bar(np.arange(num_bins) + 0.5, stage_iou_stats[stage_name])
            ax = plt.gca()
            ax.set_xticks(new_xticks)
            ax.set_xticklabels(new_xticklabels, rotation=65)
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plt.xlabel('IoU')
            plt.ylabel('amounts')
            # plt.axvline(0.5, color='k', linewidth=1.5, linestyle='--')
            # plt.axvline(0.6, color='g', linewidth=1.5, linestyle='-.')
            # plt.axvline(0.7, color='r', linewidth=1.5, linestyle='--')
            plt.title(stage_name)
            plt.show()


if __name__ == "__main__":
    main()
