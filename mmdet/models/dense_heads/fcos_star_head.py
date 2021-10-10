import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale, normal_init
from mmcv.runner import force_fp32

from mmdet.core import distance2bbox, multi_apply, multiclass_nms,StarGenerator
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
from ivipcv.ops import StarPool
import numpy as np
INF = 1e8


@HEADS.register_module()
class FCOSHead(AnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 target_means_init=[0., 0., 0., 0.],
                 target_stds_init=[1.0, 1.0, 1.0, 1.0],
                 target_means_refine=[0., 0., 0., 0.],
                 target_stds_refine=[0.1, 0.1, 0.2, 0.2],
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_init=dict(type='IoULoss', loss_weight=1.0),
                 loss_bbox_refine=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.star_generators = StarGenerator()
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox_init=loss_bbox_init,
            loss_bbox_refine=loss_bbox_refine,
            norm_cfg=norm_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.star_pooling_layers = nn.ModuleList(
            [StarPool(spatial_scale=1 / s, pool_mode='100000') for s in self.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        super().init_weights()
        normal_init(self.conv_centerness, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): Centerss for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides,self.star_pooling_layers)

    def forward_single(self, x, scale, stride,star_pooling_layers,img_metas):
        """Forward features of a single scale levle.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)

        featmap_sizes=x.size()[-2:]
        device = x[0].device

        stars_init, valid_flag = self.get_stars(featmap_sizes,
                                                img_metas, device, stride)  # edit

        reg_feat_pre_pool = self.star_pool_reg_conv_pre_init(reg_feat)
        # reg_feat_pre_pool_weight=F.softmax(self.star_pool_reg_conv_pre_init_weight(reg_feat).view(batch,channel,-1),dim=2).reshape(batch,channel,featmap_sizes[0],featmap_sizes[1])
        reg_feat_pre_pool = reg_feat * (F.sigmoid(reg_feat_pre_pool) * 1.0).exp()

        stars_init_convert = self.convertstars(torch.stack(stars_init, dim=0), img_metas)
        star_init_reg_feat_pool = star_pooling_layers(reg_feat_pre_pool, stars_init_convert)
        batch = x.size()[0]
        channel = x.size()[1]
        star_init_reg_feat_pool_all = star_init_reg_feat_pool.reshape(batch, int(channel / 1), 2, self.pool_mode_num,
                                                                      featmap_sizes[0], featmap_sizes[1])
        star_init_reg_feat_pool_row = star_init_reg_feat_pool_all[:, :, 0, :, :, :].reshape(batch, -1, featmap_sizes[0],
                                                                                            featmap_sizes[1])
        star_init_reg_feat_pool_col = star_init_reg_feat_pool_all[:, :, 1, :, :, :].reshape(batch, -1, featmap_sizes[0],
                                                                                            featmap_sizes[1])
        star_init_reg_feat_pool_row = self.relu(
            self.star_pool_reg_conv_init_row(star_init_reg_feat_pool_row) + reg_feat)
        star_init_reg_feat_pool_col = self.relu(
            self.star_pool_reg_conv_init_col(star_init_reg_feat_pool_col) + reg_feat)
        reg_out_init_row = self.star_reg_init_out_row(star_init_reg_feat_pool_row)
        reg_out_init_col = self.star_reg_init_out_col(star_init_reg_feat_pool_col)
        # star_init_reg_feat= self.relu(self.star_reg_conv_init(star_init_reg_feat_pool_sum)+reg_feat)
        reg_out_init = torch.stack((reg_out_init_row[:, 0, :, :], reg_out_init_col[:, 0, :, :],
                                    reg_out_init_row[:, 1, :, :], reg_out_init_col[:, 1, :, :]), dim=1)
        # reg_out_init = self.star_reg_init_out(reg_feat)  # szc
        #

        reg_out_init_star = self.offset_to_stars(torch.stack(stars_init, dim=0).detach(),
                                                 reg_out_init.permute(0, 2, 3, 1), img_metas,
                                                 self.target_means_init, self.target_stds_init
                                                 )
        reg_out_init_star_convert = self.convertstars(reg_out_init_star, img_metas)

        cls_feat_pre_pool = self.star_pool_cls_conv_pre(cls_feat)
        cls_feat_pre_pool = cls_feat * (F.sigmoid(cls_feat_pre_pool) * 1.0).exp()
        star_init_cls_feat_pool = star_pooling_layers(cls_feat_pre_pool, reg_out_init_star_convert)
        star_init_cls_feat_pool_all = star_init_cls_feat_pool.reshape(batch, int(channel / 1), 2, self.pool_mode_num,
                                                                      featmap_sizes[0],
                                                                      featmap_sizes[1])
        star_init_cls_feat_pool_row = star_init_cls_feat_pool_all[:, :, 0, :, :, :].reshape(batch, -1, featmap_sizes[0],
                                                                                            featmap_sizes[1])
        star_init_cls_feat_pool_col = star_init_cls_feat_pool_all[:, :, 1, :, :, :].reshape(batch, -1, featmap_sizes[0],
                                                                                            featmap_sizes[1])

        star_init_cls_feat_pool = self.relu(
            self.star_pool_cls_conv(star_init_cls_feat_pool_row + star_init_cls_feat_pool_col) + cls_feat)
        cls_out = self.star_cls_out(star_init_cls_feat_pool)

        reg_feat_refine = self.star_reg_refine_conv(reg_feat)

        reg_feat_pre_pool_refine = self.star_pool_reg_conv_pre_refine(reg_feat_refine)
        reg_feat_pre_pool_refine = reg_feat_refine * (F.sigmoid(reg_feat_pre_pool_refine) * 1.0).exp()
        star_refine_reg_feat_pool = star_pooling_layers(reg_feat_pre_pool_refine, reg_out_init_star_convert)
        star_refine_reg_feat_pool_all = star_refine_reg_feat_pool.reshape(batch, int(channel / 1), 2,
                                                                          self.pool_mode_num, featmap_sizes[0],
                                                                          featmap_sizes[1])
        star_refine_reg_feat_pool_row = star_refine_reg_feat_pool_all[:, :, 0, :, :, :].reshape(batch, -1,
                                                                                                featmap_sizes[0],
                                                                                                featmap_sizes[1])
        star_refine_reg_feat_pool_col = star_refine_reg_feat_pool_all[:, :, 1, :, :, :].reshape(batch, -1,
                                                                                                featmap_sizes[0],
                                                                                                featmap_sizes[1])
        star_refine_reg_feat_pool_row = self.relu(
            self.star_pool_reg_conv_refine_row(star_refine_reg_feat_pool_row) + reg_feat_refine)
        star_refine_reg_feat_pool_col = self.relu(
            self.star_pool_reg_conv_refine_col(star_refine_reg_feat_pool_col) + reg_feat_refine)
        reg_out_refine_row = self.star_reg_refine_out_row(star_refine_reg_feat_pool_row)
        reg_out_refine_col = self.star_reg_refine_out_col(star_refine_reg_feat_pool_col)
        reg_out_refine = torch.stack(
            (reg_out_refine_row[:, 0, :, :], reg_out_refine_col[:, 0, :, :], reg_out_refine_row[:, 1, :, :],
             reg_out_refine_col[:, 1, :, :]), dim=1)
        reg_out_refine_star = self.offset_to_stars(reg_out_init_star.detach(),
                                                   reg_out_refine.permute(0, 2, 3, 1), img_metas,
                                                   self.target_means_refine, self.target_stds_refine)  # stars
        reg_out_init_bbox = self.star2bbox([], reg_out_init_star, img_metas[0], mode='xywxyhtox1y1x2y2')
        reg_out_refine_bbox = self.star2bbox([], reg_out_refine_star, img_metas[0], mode='xywxyhtox1y1x2y2')

        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16

        # bbox_pred = scale(bbox_pred).float()
        # if self.norm_on_bbox:
        #     # bbox_pred = F.relu(bbox_pred)
        #     if not self.training:
        #         bbox_pred *= stride
        # else:
        #     bbox_pred = bbox_pred.exp()
        return cls_out, centerness, reg_out_init, reg_out_refine, stars_init, reg_out_init_star,reg_out_refine_star,reg_out_init_bbox,reg_out_refine_bbox,valid_flag
    def get_stars(self, featmap_sizes, img_metas, device,star_stride): #change,edit
        """Get stars according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: stars of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        stars = self.star_generators.grid_stars(featmap_sizes, star_stride, device)
        stars_list = [stars.clone()  for _ in range(num_imgs)]
        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            # multi_level_flags = []
            feat_h, feat_w = featmap_sizes
            h, w = img_meta['pad_shape'][:2]
            valid_feat_h = min(int(np.ceil(h / star_stride)), feat_h)
            valid_feat_w = min(int(np.ceil(w / star_stride)), feat_w)
            flags = self.star_generators.valid_flags(
                (feat_h, feat_w), (valid_feat_h, valid_feat_w), device)
            valid_flag_list.append(flags)

        return stars_list, valid_flag_list
    def offset_to_stars(self, star_init, pred_offset,img_metas,target_means,target_stds):
        """Change from star offset to star coordinate."""
        # means = pred_offset.new_tensor(self.target_means).view(1, -1).repeat(1, pred_offset.size(1) // 4)
        # stds = pred_offset.new_tensor(self.target_stds).view(1, -1).repeat(1, pred_offset.size(1) // 4)
        means=pred_offset.new_tensor(target_means)
        stds = pred_offset.new_tensor(target_stds)
        pred_offset=pred_offset*stds+means
        init_x_row = star_init[..., 0]
        init_y_row = star_init[..., 1]
        init_w = star_init[..., 2]
        init_x_col = star_init[..., 3]
        init_y_col = star_init[..., 4]
        init_h = star_init[..., 5]
        pre_off_x = pred_offset[..., 0]
        pre_off_y = pred_offset[..., 1]
        pre_off_w = pred_offset[..., 2]
        pre_off_h = pred_offset[..., 3]

        star_w = init_w * pre_off_w.exp()
        star_h = init_h * pre_off_h.exp()
        star_xc_row = init_x_row + init_w * pre_off_x # x center for row
        star_yc_row = init_y_row   # y center for row
        star_xc_col = init_x_col # x center for col
        star_yc_col = init_y_col + init_h * pre_off_y  # y center for col
        # max_shape = img_metas[0]['img_shape']
        # if max_shape is not None:
        #     star_xc_row = star_xc_row.clamp(min=0, max=max_shape[1])
        #     star_yc_row = star_yc_row.clamp(min=0, max=max_shape[1])
        #     star_w = star_w.clamp(min=0, max=max_shape[1])
        #     star_xc_col = star_xc_col.clamp(min=0, max=max_shape[0])
        #     star_yc_col = star_yc_col.clamp(min=0, max=max_shape[0])
        #     star_h = star_h.clamp(min=0, max=max_shape[0])
        stars = torch.stack([star_xc_row, star_yc_row, star_w, star_xc_col, star_yc_col,star_h], dim=-1)


        # x1 = stars_xc_row - stars[:, 2] * 0.5
        # # y1 = stars[:, 4] - stars[:, 5] * 0.5
        # x2 = stars[:, 0] + stars[:, 2] * 0.5
        # y2 = stars[:, 4] + stars[:, 5] * 0.5

        # print('=======================================================================================')
        # print('pred_offset:', pred_offset)
        # print('stars:', stars)
        return stars
    def convertstars(self, star_init,img_metas):
        """Change from star offset to star coordinate."""
        # means = pred_offset.new_tensor(self.target_means).view(1, -1).repeat(1, pred_offset.size(1) // 4)
        # stds = pred_offset.new_tensor(self.target_stds).view(1, -1).repeat(1, pred_offset.size(1) // 4)
        init_x_row = star_init[..., 0]
        init_y_row = star_init[..., 1]
        init_w = star_init[..., 2]
        init_x_col = star_init[..., 3]
        init_y_col = star_init[..., 4]
        init_h = star_init[..., 5]
        init_x1_row = init_x_row-0.5*init_w
        init_x2_row = init_x_row + 0.5 * init_w
        init_y_row=init_y_row
        init_x_col=init_x_col
        init_y1_col=init_y_col-0.5*init_h
        init_y2_col = init_y_col + 0.5 * init_h
        max_shape = img_metas[0]['img_shape']
        if max_shape is not None:
            init_x1_row = init_x1_row.clamp(min=0, max=max_shape[1])
            init_x2_row = init_x2_row.clamp(min=0, max=max_shape[1])
            init_y_row = init_y_row.clamp(min=0, max=max_shape[0])
            init_x_col = init_x_col.clamp(min=0, max=max_shape[1])
            init_y1_col = init_y1_col.clamp(min=0, max=max_shape[0])
            init_y2_col = init_y2_col.clamp(min=0, max=max_shape[0])

        stars = torch.stack([init_x1_row, init_x2_row, init_y_row, init_x_col, init_y1_col,init_y2_col], dim=-1)


        # x1 = stars_xc_row - stars[:, 2] * 0.5
        # # y1 = stars[:, 4] - stars[:, 5] * 0.5
        # x2 = stars[:, 0] + stars[:, 2] * 0.5
        # y2 = stars[:, 4] + stars[:, 5] * 0.5

        # print('=======================================================================================')
        # print('pred_offset:', pred_offset)
        # print('stars:', stars)
        return stars

    def stars_to_offset(self, predict_stars, gt_bboxes,target_means,target_stds):
        """Change from star offset to star coordinate."""

        gt_x1 = gt_bboxes[:, 0]
        gt_y1 = gt_bboxes[:, 1]
        gt_x2 = gt_bboxes[:, 2]
        gt_y2 = gt_bboxes[:, 3]
        gt_xc = (gt_x1+gt_x2)*0.5
        gt_yc = (gt_y1+gt_y2)*0.5
        gt_w = gt_x2 - gt_x1
        gt_h = gt_y2 - gt_y1
        pre_xc_row = predict_stars[:, 0]
        pre_yc_row = predict_stars[:, 1]
        pre_w = predict_stars[:, 2]
        pre_xc_col = predict_stars[:, 3]
        pre_yc_col = predict_stars[:, 4]
        pre_h = predict_stars[:, 5]
        delta_x = (gt_xc - pre_xc_row) / pre_w
        delta_y = (gt_yc - pre_yc_col) / pre_h
        delta_w = torch.log(gt_w/pre_w)
        delta_h = torch.log(gt_h/pre_h)
        delta_bboxes = torch.stack([delta_x,delta_y,delta_w,delta_h],dim=-1)
        means = delta_bboxes.new_tensor(target_means)
        stds = delta_bboxes.new_tensor(target_stds)
        delta_bboxes = delta_bboxes.sub_(means).div_(stds)
        # print('delta_bboxes:', delta_bboxes)
        return delta_bboxes
    def star2bbox(self,bboxes,stars,img_metas,mode='x1y1x2y2toxywxyh'):
        if mode =='x1y1x2y2toxywxyh':
            stars_x_w=(bboxes[...,0]+bboxes[...,2])*0.5
            stars_y_w=(bboxes[...,1]+bboxes[...,3])*0.5
            stars_x_h=stars_x_w
            stars_y_h=stars_y_w
            stars_w=bboxes[...,2]-bboxes[...,0]
            stars_h=bboxes[...,3]-bboxes[...,1]
            return torch.stack([stars_x_w,stars_y_w,stars_w,stars_x_h,stars_y_h,stars_h],dim=-1)
        if mode == 'xywxyhtox1y1x2y2':
            x1=stars[...,0]-stars[...,2]*0.5
            y1=stars[...,4]-stars[...,5]*0.5
            x2=stars[...,0]+stars[...,2]*0.5
            y2 = stars[..., 4] + stars[..., 5] * 0.5
            y_row=stars[...,1]
            x_col = stars[..., 3]
            # if ((x_col<x1) or (x_col>x2) or (y_row<y1) or (y_row>y2)):
            #     x1=torch.zeros_like(x1)
            #     x2 = torch.zeros_like(x2)
            #     y1 = torch.zeros_like(y1)
            #     y2 = torch.zeros_like(y2)
            # x1=torch.where(x1<=x_col,x1,torch.zeros_like(x1))
            # x1 = torch.where(x1 <= x_col, x1, torch.zeros_like(x1))
            max_shape=img_metas['img_shape']
            if max_shape is not None:
                x1 = x1.clamp(min=0, max=max_shape[1])
                y1 = y1.clamp(min=0, max=max_shape[0])
                x2 = x2.clamp(min=0, max=max_shape[1])
                y2 = y2.clamp(min=0, max=max_shape[0])
            bbox=torch.stack([x1,y1,x2,y2],dim=-1)
            return bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             centernesses,
             reg_out_init,
             reg_out_refine,
             stars_init,
             reg_out_init_star,
             reg_out_refine_star,
             reg_out_init_bbox,
             reg_out_refine_bbox,
             valid_flag,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): Centerss for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(reg_out_init) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
        #                                    bbox_preds[0].device)
        candidate_list = stars_init
        labels, bbox_targets = self.get_targets(candidate_list,gt_bboxes,
                                                gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.reshape(-1, 4)
            for bbox_pred in reg_out_refine_bbox
        ]

        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            # pos_points = flatten_points[pos_inds]
            # pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            # pos_decoded_target_preds = distance2bbox(pos_points,
            #                                          pos_bbox_targets)
            # centerness weighted iou loss
            loss_bbox = self.loss_bbox(
                pos_bbox_preds,
                pos_bbox_targets,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum())
            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(
                cls_score_list, bbox_pred_list, centerness_pred_list,
                mlvl_points, img_shape, scale_factor, cfg, rescale, with_nms)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           centernesses,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(
                mlvl_bboxes,
                mlvl_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=mlvl_centerness)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_centerness

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_targets(self, stars, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(stars) == len(self.regress_ranges)
        num_levels = len(stars)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            stars[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                stars[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(stars, dim=0)

        # the number of points per img, per lvl
        num_stars = [center.size(0) for center in stars]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_stars)

        # split to per img, per level
        labels_list = [labels.split(num_stars, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_stars, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, stars, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_stars = stars.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_stars,), self.num_classes), \
                   gt_bboxes.new_zeros((num_stars, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_stars, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_stars, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_stars, num_gts, 4)

        xs = (stars[:, 0]+stars[:, 1])*0.5
        ys = (stars[:, 4]+stars[:, 5])*0.5

        xs = xs[:, None].expand(num_stars, num_gts)
        ys = ys[:, None].expand(num_stars, num_gts)

        # left = xs - gt_bboxes[..., 0]
        # right = gt_bboxes[..., 2] - xs
        # top = ys - gt_bboxes[..., 1]
        # bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((gt_bboxes[..., 0], gt_bboxes[..., 1], gt_bboxes[..., 2], gt_bboxes[..., 3]), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_stars), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
