import numpy as np
import torch

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class DeltaXYWHXYWHBBoxCoder(BaseBBoxCoder):
    """Delta XYWHXYWH BBox coder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes bbox (x1, y1, x2, y2) into delta (dx, dy, dw, dh) and
    decodes delta (dx, dy, dw, dh) back to original bbox (x1, y1, x2, y2).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
    """

    def __init__(self,
                 target_means=(0., 0., 0., 0.,0.,0.),
                 target_stds=(1., 1., 1., 1.,1.,1.),
                 clip_border=True):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds
        self.clip_border = clip_border

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (torch.Tensor): Basic boxes.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            max_shape (tuple[int], optional): Maximum shape of boxes.
                Defaults to None.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            torch.Tensor: Decoded boxes.
        """

        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta2bbox(bboxes, pred_bboxes, self.means, self.stds,
                                    max_shape, wh_ratio_clip, self.clip_border)

        return decoded_bboxes


def bbox2delta(proposals, gt, means=(0., 0., 0., 0., 0., 0.), stds=(1., 1., 1., 1., 1., 1.)):
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of x, y, w, h of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of :func:`delta2bbox`.

    Args:
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 4)
        gt (Tensor): Gt bboxes to be used as base, shape (N, ..., 4)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates

    Returns:
        Tensor: deltas with shape (N, 4), where columns represent dx, dy,
            dw, dh.
    """
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    pxw = proposals[..., 0]
    pyw = proposals[..., 1]
    pw = proposals[..., 2]
    pxh = proposals[..., 3]
    pyh = proposals[..., 4]
    ph = proposals[..., 5]

    gxw = gt[..., 0]
    gyw = gt[..., 1]
    gw = gt[..., 2]
    gxh = gt[..., 3]
    gyh = gt[..., 4]
    gh = gt[..., 5]

    dxw = (gxw - pxw) / pw
    dyw = (gyw - pyw) / ph
    dw = torch.log(gw / pw)
    dxh = (gxh - pxh) / pw
    dyh = (gyh - pyh) / ph
    dh = torch.log(gh / ph)
    deltas = torch.stack([dxw, dyw, dw, dxh, dyh, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2bbox(stars,
               deltas,
               means=(0., 0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               clip_border=True):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (N, 4 * num_classes). Note N = num_anchors * W * H when
            rois is a grid of anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_shape (tuple[int, int]): Maximum bounds for boxes. specifies (H, W)
        wh_ratio_clip (float): Maximum aspect ratio for boxes.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.

    Returns:
        Tensor: Boxes with shape (N, 4), where columns represent
            tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    """
    means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 6)
    stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 6)
    denorm_deltas = deltas * stds + means
    dxw = denorm_deltas[:, 0::6]
    dyw = denorm_deltas[:, 1::6]
    dw = denorm_deltas[:, 2::6]
    dxh = denorm_deltas[:, 3::6]
    dyh = denorm_deltas[:, 4::6]
    dh = denorm_deltas[:, 5::6]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    pxw = stars[:,0].unsqueeze(1).expand_as(dxw)
    pyw = stars[:, 1] .unsqueeze(1).expand_as(dyw)
    pw = stars[:, 2].unsqueeze(1).expand_as(dw)
    pxh = stars[:, 3].unsqueeze(1).expand_as(dxh)
    pyh = stars[:, 4].unsqueeze(1).expand_as(dyh)
    ph = stars[:, 5].unsqueeze(1).expand_as(dh)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gxw = pxw + pw * dxw
    gyw = pyw + ph * dyw
    gxh = pxh + pw * dxh
    gyh = pyh + ph * dyh
    bboxes = torch.stack([gxw, gyw, gw, gxh, gyh, gh], dim=-1).view(deltas.size())
    return bboxes
