import torch

from ..builder import BBOX_ASSIGNERS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class CrossAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each point.

    Each proposals will be assigned with `0`, or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    """

    def __init__(self, scale=4, pos_num=3):
        self.scale = scale
        self.pos_num = pos_num

    def assign(self, cross, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to points.

        This method assign a gt bbox to every points set, each points set
        will be assigned with  the background_label (-1), or a label number.
        -1 is background, and semi-positive number is the index (0-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every points to the background_label (-1)
        2. A point is assigned to some gt bbox if
            (i) the point is within the k closest points to the gt bbox
            (ii) the distance between this point and the gt is smaller than
                other gt bboxes

        Args:
            points (Tensor): points to be assigned, shape(n, 3) while last
                dimension stands for (x, y, stride).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
                NOTE: currently unused.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_cross = cross.shape[0]
        num_gts = gt_bboxes.shape[0]

        if num_gts == 0 or num_cross == 0:
            # If no truth assign everything to the background
            assigned_gt_inds = cross.new_full((num_cross, ),
                                               0,
                                               dtype=torch.long)
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = cross.new_full((num_cross, ),
                                                  -1,
                                                  dtype=torch.long)
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        cross_xy=cross[:,:2]
        cross_w = cross[:, 2]
        cross_stride=cross_w*0.25
        cross_lvl = torch.log2(
            cross_stride).int()  # [3...,4...,5...,6...,7...]
        lvl_min, lvl_max = cross_lvl.min(), cross_lvl.max()

        # assign gt box
        gt_bboxes_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
        gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min=1e-6)
        scale = self.scale
        gt_bboxes_lvl = ((torch.log2(gt_bboxes_wh[:, 0] / scale) +
                          torch.log2(gt_bboxes_wh[:, 1] / scale)) / 2).int()
        gt_bboxes_lvl = torch.clamp(gt_bboxes_lvl, min=lvl_min, max=lvl_max)

        # stores the assigned gt index of each point
        assigned_gt_inds = cross.new_zeros((num_cross, ), dtype=torch.long)
        # stores the assigned gt dist (to this point) of each point
        assigned_gt_dist = cross.new_full((num_cross, ), float('inf'))
        cross_range = torch.arange(cross.shape[0])

        for idx in range(num_gts):
            gt_lvl = gt_bboxes_lvl[idx]
            # get the index of points in this level
            lvl_idx = gt_lvl == cross_lvl
            cross_index = cross_range[lvl_idx]
            # get the points in this level
            lvl_cross = cross_xy[lvl_idx, :]
            lvl_cross_w=cross_w[lvl_idx]
            # get the center point of gt
            gt_cross = gt_bboxes_xy[[idx], :]
            # get width and height of gt
            gt_wh = gt_bboxes_wh[[idx], :]
            # compute the distance between gt center and
            #   all points in this level
            cross_gt_dist = ((lvl_cross - gt_cross) / gt_wh).norm(dim=1)

            # find the nearest k points to gt center in this level

            min_dist, min_dist_index = torch.topk( #orig,qhq
                cross_gt_dist, self.pos_num, largest=False)

            # the index of nearest k points to gt center in this level
            min_dist_cross_index = cross_index[min_dist_index]
            # The less_than_recorded_index stores the index
            #   of min_dist that is less then the assigned_gt_dist. Where
            #   assigned_gt_dist stores the dist from previous assigned gt
            #   (if exist) to each point.
            less_than_recorded_index = min_dist < assigned_gt_dist[
                min_dist_cross_index]
            # The min_dist_points_index stores the index of points satisfy:
            #   (1) it is k nearest to current gt center in this level.
            #   (2) it is closer to current gt center than other gt center.
            min_dist_cross_index = min_dist_cross_index[
                less_than_recorded_index]

            assigned_gt_inds[min_dist_cross_index] = idx + 1
            assigned_gt_dist[min_dist_cross_index] = min_dist[
                less_than_recorded_index]
            # assigned_gt_inds[pos_cross_index] = idx + 1
            # assigned_gt_dist[pos_cross_index] = min_dist[
            #     less_than_recorded_index]

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_cross, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
