import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms,CrossGenerator,multiclass_nms_cross,
                        reduce_mean, unmap)
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead
from ivipcv.ops import CrossPool
import numpy as np
import torch.nn.functional as F

EPS = 1e-12
INF = 1e8

@HEADS.register_module()
class CrossHead(AnchorHead):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection via
    Adaptive Training Sample Selection.

    ATSS head structure is similar with FCOS, however ATSS use anchor boxes
    and assign label by Adaptive Training Sample Selection instead max-iou.

    https://arxiv.org/abs/1912.02424
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 center_sampling=True,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 target_means_init=[0., 0., 0., 0.],
                 target_stds_init=[1.0, 1.0, 1.0, 1.0],
                 target_means_refine=[0., 0., 0., 0.],
                 target_stds_refine=[0.1, 0.1, 0.2, 0.2],
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox_init=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_bbox_refine=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        # print (norm_cfg)
        self.cross_generators = CrossGenerator()
        self.regress_ranges = regress_ranges
        self.target_means_refine = target_means_refine
        self.target_stds_refine = target_stds_refine
        super(CrossHead, self).__init__(num_classes, in_channels, **kwargs)
        self.target_means_init = target_means_init
        self.target_stds_init = target_stds_init
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox=norm_on_bbox
        self.sampling = False
        if self.train_cfg:
           
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_bbox_init = build_loss(loss_bbox_init)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)
        self.loss_dist_rank=nn.MarginRankingLoss()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.h_pool = nn.AdaptiveAvgPool2d((1, None))
        self.w_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.h_conv = nn.Conv2d(self.feat_channels, self.feat_channels, (1, 3), 1, (0, 1), bias=True)
        self.w_conv = nn.Conv2d(self.feat_channels, self.feat_channels, (3, 1), 1, (1, 0), bias=True)
        self.hw_conv = ConvModule(
            self.feat_channels,
            self.feat_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)
      
        self.atss_centerness = nn.Conv2d(
            self.feat_channels, self.num_anchors * 1, 3, padding=1)

        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.anchor_generator.strides])
        self.strides = [s[0] for s in self.anchor_generator.strides]
        self.pool_mode_num = 1
        self.cross_pooling_layers = nn.ModuleList(
            [CrossPool(spatial_scale=1 / s, pool_mode='100000') for s in self.strides])
       

        self.cross_pool_cls_conv_pre = nn.Conv2d(self.feat_channels * 1, int(self.feat_channels / 1), 3, 1, 1)
      
        self.cross_pool_reg_conv_init_row = nn.Conv2d(self.feat_channels * self.pool_mode_num * 1,
                                                     self.feat_channels, 3, 1, 1)
        
        self.cross_pool_reg_conv_init_col = nn.Conv2d(self.feat_channels * self.pool_mode_num * 1,
                                                     self.feat_channels, 3, 1, 1)
       
        self.cross_pool_cls_conv = nn.Conv2d(self.feat_channels * self.pool_mode_num * 2, self.feat_channels,
                                            3, 1, 1)
      
        self.cross_cls_out = nn.Conv2d(self.feat_channels * 1,
                                      self.cls_out_channels, 1, 1, 0)
        self.cross_reg_init_out_row = nn.Conv2d(self.feat_channels,
                                               2, 1, 1, 0)
        self.cross_reg_init_out_col = nn.Conv2d(self.feat_channels,
                                               2, 1, 1, 0)
      
        self.cross_pool_reg_conv_refine_row = nn.Conv2d(self.feat_channels * self.pool_mode_num * 1,
                                                       self.feat_channels, 3, 1, 1)
       
        self.cross_pool_reg_conv_refine_col = nn.Conv2d(self.feat_channels * self.pool_mode_num * 1,
                                                       self.feat_channels, 3, 1, 1)
        

        self.cross_reg_refine_conv = nn.Conv2d(self.feat_channels * 1,
                                              self.feat_channels, 3, 1, 1)
       
        self.cross_reg_refine_out_row = nn.Conv2d(self.feat_channels,
                                                 2, 1, 1, 0)
        self.cross_reg_refine_out_col = nn.Conv2d(self.feat_channels,
                                                 2, 1, 1, 0)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.cross_cls_out, std=0.01, bias=bias_cls)
        normal_init(self.cross_reg_init_out_row, std=0.01)
        normal_init(self.cross_reg_init_out_col, std=0.01)
        normal_init(self.atss_centerness, std=0.01)

    def forward(self, feats, img_metas):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        return multi_apply(self.forward_single, feats, img_metas, self.cross_pooling_layers, self.strides, self.scales)

    def forward_single(self, x, img_metas, cross_pooling_layers, stride, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
                centerness (Tensor): Centerness for a single scale level, the
                    channel number is (N, num_anchors * 1, H, W).
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        batch = x.size()[0]
        channel = x.size()[1]
        featmap_sizes = x.size()[-2:]
        device = x[0].device

        reg_feat_hw = self.hw_conv(x)
        reg_feat_pre_h_pool = self.h_pool(reg_feat_hw)
        reg_feat_pre_w_pool = self.w_pool(reg_feat_hw)
        reg_feat_h = F.interpolate(self.h_conv(reg_feat_pre_h_pool), featmap_sizes, mode='bilinear', align_corners=True)
        reg_feat_w = F.interpolate(self.w_conv(reg_feat_pre_w_pool), featmap_sizes, mode='bilinear', align_corners=True)
        reg_feat_pre_pool_hw = reg_feat_h + reg_feat_w

        # crosspooling,cross feat

        cross_init_list, valid_flag = self.get_cross(featmap_sizes,
                                                     img_metas, device, stride)  
        cross_init = torch.stack(cross_init_list, dim=0)


        cross_init_reg_feat_pool = cross_pooling_layers(reg_feat, cross_init)
        cross_init_reg_feat_pool_all = cross_init_reg_feat_pool.reshape(batch, int(channel / 1), 2, self.pool_mode_num,
                                                                      featmap_sizes[0], featmap_sizes[1])
        cross_init_reg_feat_pool_row = cross_init_reg_feat_pool_all[:, :, 0, :, :, :].reshape(batch, -1, featmap_sizes[0],
                                                                                            featmap_sizes[1])
        cross_init_reg_feat_pool_col = cross_init_reg_feat_pool_all[:, :, 1, :, :, :].reshape(batch, -1, featmap_sizes[0],
                                                                                            featmap_sizes[1])
        cross_init_reg_feat_pool_row = self.relu(
            self.cross_pool_reg_conv_init_row(cross_init_reg_feat_pool_row) + reg_feat)
        cross_init_reg_feat_pool_col = self.relu(
            self.cross_pool_reg_conv_init_col(cross_init_reg_feat_pool_col) + reg_feat)

        reg_out_init_row = scale(self.cross_reg_init_out_row(cross_init_reg_feat_pool_row))
        reg_out_init_col = scale(self.cross_reg_init_out_col(cross_init_reg_feat_pool_col))
        reg_out_init = torch.stack((reg_out_init_row[:, 0, :, :], reg_out_init_col[:, 0, :, :],
                                    reg_out_init_row[:, 1, :, :], reg_out_init_col[:, 1, :, :]), dim=1)

        reg_out_init_cross = self.offset_to_cross(cross_init.detach(),
                                                 reg_out_init.permute(0, 2, 3, 1), img_metas,
                                                 self.target_means_init, self.target_stds_init
                                                 )


        cls_feat_pre_pool = self.cross_pool_cls_conv_pre(cls_feat + reg_feat_pre_pool_hw)
        cls_feat_pre_pool = cls_feat_pre_pool * (F.sigmoid(cls_feat_pre_pool) * 1.0).exp()
        cross_init_cls_feat_pool = cross_pooling_layers(cls_feat_pre_pool, reg_out_init_cross)
        cross_init_cls_feat_pool_all = cross_init_cls_feat_pool.reshape(batch, int(channel / 1), 2, self.pool_mode_num,
                                                                      featmap_sizes[0],
                                                                      featmap_sizes[1])
        cross_init_cls_feat_pool_row = cross_init_cls_feat_pool_all[:, :, 0, :, :, :].reshape(batch, -1, featmap_sizes[0],
                                                                                            featmap_sizes[1])
        cross_init_cls_feat_pool_col = cross_init_cls_feat_pool_all[:, :, 1, :, :, :].reshape(batch, -1, featmap_sizes[0],
                                                                                            featmap_sizes[1])

        cross_init_cls_feat_pool = self.relu(
            self.cross_pool_cls_conv(
                torch.cat((cross_init_cls_feat_pool_row, cross_init_cls_feat_pool_col), dim=1)) + cls_feat)
        cls_out = self.cross_cls_out(cross_init_cls_feat_pool)

        reg_feat_refine = self.cross_reg_refine_conv(reg_feat + reg_feat_pre_pool_hw)
        reg_feat_pre_pool_refine = reg_feat_refine * (F.sigmoid(reg_feat_refine) * 1.0).exp()
        cross_refine_reg_feat_pool = cross_pooling_layers(reg_feat_pre_pool_refine, reg_out_init_cross)
        cross_refine_reg_feat_pool_all = cross_refine_reg_feat_pool.reshape(batch, int(channel / 1), 2,
                                                                          self.pool_mode_num, featmap_sizes[0],
                                                                          featmap_sizes[1])
        cross_refine_reg_feat_pool_row = cross_refine_reg_feat_pool_all[:, :, 0, :, :, :].reshape(batch, -1,
                                                                                                featmap_sizes[0],
                                                                                                featmap_sizes[1])
        cross_refine_reg_feat_pool_col = cross_refine_reg_feat_pool_all[:, :, 1, :, :, :].reshape(batch, -1,
                                                                                                featmap_sizes[0],
                                                                                                featmap_sizes[1])

        cross_refine_reg_feat_pool_row = self.relu(
            self.cross_pool_reg_conv_refine_row(cross_refine_reg_feat_pool_row) + reg_feat)
        cross_refine_reg_feat_pool_col = self.relu(
            self.cross_pool_reg_conv_refine_col(cross_refine_reg_feat_pool_col) + reg_feat)
        reg_out_refine_row = self.cross_reg_refine_out_row(cross_refine_reg_feat_pool_row)
        reg_out_refine_col = self.cross_reg_refine_out_col(cross_refine_reg_feat_pool_col)
        reg_out_refine = torch.stack(
            (reg_out_refine_row[:, 0, :, :], reg_out_refine_col[:, 0, :, :], reg_out_refine_row[:, 1, :, :],
             reg_out_refine_col[:, 1, :, :]), dim=1)
        reg_out_refine_cross = self.offset_to_cross(reg_out_init_cross.detach(),
                                                   reg_out_refine.permute(0, 2, 3, 1), img_metas,
                                                   self.target_means_refine, self.target_stds_refine)  # cross
        centerness = self.atss_centerness(reg_feat)
        cross_init_bbox = self.cross2bbox([], cross_init, img_metas[0], mode='x1x2yxy1y2tox1y1x2y2', clip=False)
        reg_out_init_bbox = self.cross2bbox([], reg_out_init_cross, img_metas[0], mode='x1x2yxy1y2tox1y1x2y2', clip=True)
        reg_out_refine_bbox = self.cross2bbox([], reg_out_refine_cross, img_metas[0], mode='x1x2yxy1y2tox1y1x2y2',
                                             clip=True)
        return cls_out, reg_out_init, reg_out_refine, centerness, cross_init_bbox, reg_out_init_bbox, reg_out_refine_bbox,reg_out_refine_cross, valid_flag

    def loss_single(self, anchors, cls_score, reg_out_init_bbox,reg_out_refine_bbox, centerness, labels_init,

                    bbox_targets_init, labels_refine,
                    label_weights_refine, bbox_weights_refine, bbox_targets_refine, num_total_samples_init,num_total_samples_refine):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            num_total_samples (int): Number os positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()

        reg_out_init_bbox=reg_out_init_bbox.reshape(-1,4)
        reg_out_refine_bbox = reg_out_refine_bbox.reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets_init = bbox_targets_init.reshape(-1, 4)
        bbox_targets_refine = bbox_targets_refine.reshape(-1, 4)
        labels_init = labels_init.reshape(-1)

        labels_refine = labels_refine.reshape(-1)
        label_weights_refine = label_weights_refine.reshape(-1)

        bbox_weights_refine = bbox_weights_refine.reshape(-1, 4)
        # classification loss
        loss_cls = self.loss_cls(
            cls_score, labels_refine, label_weights_refine, avg_factor=num_total_samples_refine)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds_init = ((labels_init >= 0)
                    & (labels_init < bg_class_ind)).nonzero().squeeze(1)
        pos_inds_refine = ((labels_refine >= 0)
                         & (labels_refine < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds_init) > 0:
            pos_bbox_targets_init = bbox_targets_init[pos_inds_init]
            pos_bbox_pred_init = reg_out_init_bbox[pos_inds_init]

            pos_anchors = anchors[pos_inds_init]
            anchor_xc=(pos_anchors[:,0]+pos_anchors[:,2])*0.5
            anchor_yc = (pos_anchors[:, 1] + pos_anchors[:, 3]) * 0.5
            pred_x1=pos_bbox_pred_init[:,0]
            pred_y1 = pos_bbox_pred_init[:, 1]
            pred_x2 = pos_bbox_pred_init[:, 2]
            pred_y2 = pos_bbox_pred_init[:, 3]


            pos_centerness = centerness[pos_inds_init]

            centerness_targets_init = self.centerness_target(
                pos_anchors, pos_bbox_targets_init)


            # regression loss
            loss_bbox_init = self.loss_bbox_init(
                pos_bbox_pred_init,
                pos_bbox_targets_init,
                weight=centerness_targets_init,
                avg_factor=1.0)
            label_dist_init=torch.ones_like(pred_x1)
            loss_bbox_dist_init=0.1*(self.loss_dist_rank(anchor_xc,pred_x1,label_dist_init)+self.loss_dist_rank(pred_x2,anchor_xc,label_dist_init)\
                           +self.loss_dist_rank(anchor_yc,pred_y1,label_dist_init)+self.loss_dist_rank(pred_y2,anchor_yc,label_dist_init))

            # centerness loss
            loss_centerness = self.loss_centerness(
                pos_centerness,
                centerness_targets_init,
                avg_factor=num_total_samples_init)
        else:
            loss_bbox_init = reg_out_init_bbox.sum() * 0
            loss_bbox_dist_init = reg_out_init_bbox.sum() * 0

            loss_centerness = centerness.sum() * 0
            centerness_targets_init = bbox_targets_init.new_tensor(0.)
        if len(pos_inds_refine)>0:
            pos_bbox_targets_refine = bbox_targets_refine[pos_inds_refine]
            pos_bbox_pred_refine = reg_out_refine_bbox[pos_inds_refine]
            pos_bbox_weights_refine = bbox_weights_refine[pos_inds_refine][:, 0]
            pos_bbox_inint_bbox = reg_out_init_bbox[pos_inds_refine]

            bbox_init_xc = (pos_bbox_inint_bbox[:, 0] + pos_bbox_inint_bbox[:, 2]) * 0.5
            bbox_init_yc = (pos_bbox_inint_bbox[:, 1] + pos_bbox_inint_bbox[:, 3]) * 0.5
            pred_refine_x1 = pos_bbox_pred_refine[:, 0]
            pred_refine_y1 = pos_bbox_pred_refine[:, 1]
            pred_refine_x2 = pos_bbox_pred_refine[:, 2]
            pred_refine_y2 = pos_bbox_pred_refine[:, 3]
            label_dist_refine = torch.ones_like(pred_refine_x1)
            centerness_targets_refine = self.centerness_target(
                pos_bbox_inint_bbox, pos_bbox_targets_refine)
            loss_bbox_refine = self.loss_bbox_refine(
                pos_bbox_pred_refine,
                pos_bbox_targets_refine,
                weight=pos_bbox_weights_refine,
                avg_factor=1.0)
            loss_bbox_dist_refine = 0.1 * (
                        self.loss_dist_rank(bbox_init_xc, pred_refine_x1, label_dist_refine) + self.loss_dist_rank(pred_refine_x2, bbox_init_xc,
                                                                                                  label_dist_refine) \
                        + self.loss_dist_rank(bbox_init_yc, pred_refine_y1, label_dist_refine) + self.loss_dist_rank(pred_refine_y2, bbox_init_yc,
                                                                                                    label_dist_refine))
        else:
            loss_bbox_refine = reg_out_refine_bbox.sum() * 0
            pos_bbox_weights_refine = bbox_targets_refine.new_tensor(0.)
            loss_bbox_dist_refine = reg_out_init_bbox.sum() * 0


        return loss_cls, loss_bbox_init, loss_bbox_refine, loss_centerness,loss_bbox_dist_init,loss_bbox_dist_refine, centerness_targets_init.sum(),pos_bbox_weights_refine.sum()

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds_init,
             bbox_preds_refine,
             centernesses,
             cross_init_bbox,
             reg_out_init_bbox,
             reg_out_refine_bbox,
             reg_out_refine_cross,
             valid_flag_list,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        all_level_points = self.get_points(featmap_sizes, bbox_preds_init[0].dtype,
                                           bbox_preds_init[0].device)
        anchor_list = cross_init_bbox
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        #init stage
        cls_reg_targets_init = self.get_targets_fcos(
            all_level_points,
            gt_bboxes,
            gt_labels)

        if cls_reg_targets_init is None:
            return None

        (labels_list_init, bbox_targets_list_init,num_total_samples_init) = cls_reg_targets_init

        num_total_samples_init = max(num_total_samples_init, 1.0)

        # refine stage
        cls_reg_targets_refine = self.get_targets(
            reg_out_init_bbox,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets_refine is None:
            return None

        (anchor_list_refine, labels_list_refine, label_weights_list_refine, bbox_targets_list_refine,
         bbox_weights_list_refine, num_total_pos_refine, num_total_neg_refine) = cls_reg_targets_refine

        num_total_samples_refine = reduce_mean(
            torch.tensor(num_total_pos_refine).cuda()).item()
        num_total_samples_refine = max(num_total_samples_refine, 1.0)
        losses_cls, losses_bbox_init, losses_bbox_refine, loss_centerness,loss_bbox_dist_init,loss_bbox_dist_refine,\
            bbox_avg_factor_init, bbox_avg_factor_refine = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                reg_out_init_bbox,
                reg_out_refine_bbox,
                centernesses,
                labels_list_init,

                bbox_targets_list_init,
                labels_list_refine,
                label_weights_list_refine,
                bbox_weights_list_refine,
                bbox_targets_list_refine,
                num_total_samples_init=num_total_samples_init,
                num_total_samples_refine=num_total_samples_refine,
        )

        bbox_avg_factor_init = sum(bbox_avg_factor_init)
        bbox_avg_factor_init = reduce_mean(bbox_avg_factor_init).item()
        if bbox_avg_factor_init < EPS:
            bbox_avg_factor_init = 1
        losses_bbox_init = list(map(lambda x: x / bbox_avg_factor_init, losses_bbox_init))
        loss_bbox_dist_init =list(map(lambda x: x / bbox_avg_factor_init, loss_bbox_dist_init))
        bbox_avg_factor_refine = sum(bbox_avg_factor_refine)
        bbox_avg_factor_refine = reduce_mean(bbox_avg_factor_refine).item()
        if bbox_avg_factor_refine < EPS:
            bbox_avg_factor_refine = 1
        losses_bbox_refine = list(map(lambda x: x / bbox_avg_factor_refine, losses_bbox_refine))
        loss_bbox_dist_refine = list(map(lambda x: x / bbox_avg_factor_refine, loss_bbox_dist_refine))

        return dict(
            loss_cls=losses_cls,
            loss_bbox_init=losses_bbox_init,
            loss_bbox_refine=losses_bbox_refine,
            loss_centerness=loss_centerness,
            loss_bbox_dist_init=loss_bbox_dist_init,
            loss_bbox_dist_refine=loss_bbox_dist_refine,
        )

    def centerness_target(self, anchors, bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        # gts = self.bbox_coder.decode(anchors, bbox_targets)
        gts=bbox_targets
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        centerness = torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *
            (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness



    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds_init,
                   bbox_preds_refine,
                   centernesses,
                   cross_init_bbox,
                   reg_out_init_bbox,
                   reg_out_refine_bbox,
                   reg_out_refine_cross,
                   valid_flags,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_anchors * 1, H, W).
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
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds_init)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]

            bbox_pred_list = [
                reg_out_refine_bbox[i][img_id].permute(2,0,1).detach() for i in range(num_levels)
            ]
            cross_pred_list=[
                reg_out_refine_cross[i][img_id].permute(2,0,1).detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,cross_pred_list,
                                                centerness_pred_list,
                                                mlvl_anchors, img_shape,
                                                scale_factor, cfg, rescale,
                                                with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           cross_preds,
                           centernesses,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single
                scale level with shape (num_anchors * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_anchors * 1, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
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
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_cross = []
        mlvl_centerness = []
        for cls_score, bbox_pred,cross_pred, centerness, anchors in zip(
                cls_scores, bbox_preds,cross_preds, centernesses, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            cross_pred = cross_pred.permute(1, 2, 0).reshape(-1, 6)
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:

                max_scores, _ = (scores ).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                cross_pred = cross_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]


            bboxes=bbox_pred
            cross=cross_pred
            mlvl_bboxes.append(bboxes)
            mlvl_cross.append(cross)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_cross = torch.cat(mlvl_cross)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            scale_factor_cross=np.array([scale_factor[0],scale_factor[0],scale_factor[1],scale_factor[0],scale_factor[1],scale_factor[1]])
            mlvl_cross /= mlvl_cross.new_tensor(scale_factor_cross)
        mlvl_scores = torch.cat(mlvl_scores)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)

        if with_nms:
            det_bboxes, det_labels,det_cross = multiclass_nms_cross(
                mlvl_bboxes,
                mlvl_scores,
                mlvl_cross,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=mlvl_centerness,
                return_inds=False)
            # det_cross=mlvl_cross[det_keep]
            return det_bboxes, det_labels,det_cross
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_centerness

    def get_targets(self,
                    anchor_list,
                    valid_flag,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for ATSS head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)


        proposals_list = []
        valid_flag_list = []
        for i_img in range(len(img_metas)):
            proposals_lvl_list = []
            valid_flag_lvl_list = []
            for i_lvl in range(len(anchor_list)):
                proposals_lvl = anchor_list[i_lvl][i_img].view(-1, 4)
                proposals_lvl_list.append(proposals_lvl)
                valid_flag_lvl_list.append(valid_flag[i_lvl][i_img])
            proposals_list.append(proposals_lvl_list)
            valid_flag_list.append(valid_flag_lvl_list)

        assert len(proposals_list) == len(valid_flag_list) == num_imgs
        # points number of multi levels
        num_level_anchors = [cross.size(0) for cross in proposals_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])



        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             proposals_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)
    def get_targets_fcos(self, points, gt_bboxes_list, gt_labels_list):
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
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)

        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single_fcos,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
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
        flatten_labels = torch.cat(concat_lvl_labels)
        # flatten_bbox_targets = torch.cat(concat_lvl_bbox_targets)
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)

        return concat_lvl_labels, concat_lvl_bbox_targets,num_pos
    def _get_target_single_fcos(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        #
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets_dist = torch.stack((left, top, right, bottom), -1)
        bbox_targets=gt_bboxes

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
            inside_gt_bbox_mask = bbox_targets_dist.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets_dist.max(-1)[0]
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
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets
    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4)
                pos_inds (Tensor): Indices of postive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
    def get_cross(self, featmap_sizes, img_metas, device,cross_stride): #change,edit
        """Get cross according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: cross of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        cross = self.cross_generators.grid_cross(featmap_sizes, cross_stride, device)
        cross_list = [cross.clone()  for _ in range(num_imgs)]
        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            # multi_level_flags = []
            feat_h, feat_w = featmap_sizes
            h, w = img_meta['pad_shape'][:2]
            valid_feat_h = min(int(np.ceil(h / cross_stride)), feat_h)
            valid_feat_w = min(int(np.ceil(w / cross_stride)), feat_w)
            flags = self.cross_generators.valid_flags(
                (feat_h, feat_w), (valid_feat_h, valid_feat_w), device)
            valid_flag_list.append(flags)

        return cross_list, valid_flag_list

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""

        h, w = featmap_size
        x_range = torch.arange(w, dtype=dtype, device=device)
        y_range = torch.arange(h, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1)
        return points
    def get_points(self, featmap_sizes, dtype, device, flatten=False):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self._get_points_single(featmap_sizes[i], self.strides[i],
                                        dtype, device, flatten))
        return mlvl_points
    def offset_to_cross(self, cross_init, pred_offset,img_metas,target_means,target_stds):
        """Change from cross offset to cross coordinate."""

        means=pred_offset.new_tensor(target_means)
        stds = pred_offset.new_tensor(target_stds)
        pred_offset=pred_offset*stds+means
        init_x1_row = cross_init[..., 0]
        init_x2_row = cross_init[..., 1]
        init_y_row = cross_init[..., 2]
        init_x_col = cross_init[..., 3]
        init_y1_col = cross_init[..., 4]
        init_y2_col = cross_init[..., 5]



        init_xc_row = (init_x1_row+init_x2_row)*0.5

        init_w = init_x2_row-init_x1_row

        init_yc_col = (init_y1_col+init_y2_col)*0.5
        init_h = init_y2_col-init_y1_col

        pre_off_x = pred_offset[..., 0]
        pre_off_y = pred_offset[..., 1]
        pre_off_w = pred_offset[..., 2]
        pre_off_h = pred_offset[..., 3]

        cross_w = init_w * pre_off_w.exp()
        cross_h = init_h * pre_off_h.exp()
        cross_xc_row = init_xc_row + init_w * pre_off_x # x center for row
        cross_yc_col = init_yc_col + init_h * pre_off_y  # y center for col

        cross_x1_row=cross_xc_row-cross_w*0.5
        cross_x2_row=cross_xc_row+cross_w*0.5
        cross_y1_col = cross_yc_col - cross_h * 0.5
        cross_y2_col = cross_yc_col + cross_h * 0.5
        cross_y_row = init_y_row
        cross_x_col = init_x_col
        cross = torch.stack([cross_x1_row, cross_x2_row, cross_y_row,  cross_x_col, cross_y1_col, cross_y2_col], dim=-1)
        return cross
    def convertcross(self, anchor_init,img_metas):
        """Change from cross offset to cross coordinate."""
        init_x1_row = anchor_init[..., 0]
        init_y1_col = anchor_init[..., 1]
        init_x2_row = anchor_init[..., 2]
        init_y2_col = anchor_init[..., 3]
        init_x_col = (init_x1_row + init_x2_row)*0.5
        init_y_row = (init_y1_col + init_y2_col) * 0.5
        cross = torch.stack([init_x1_row, init_x2_row, init_y_row, init_x_col, init_y1_col,init_y2_col], dim=-1)
        return cross

    def cross_to_offset(self, predict_cross, gt_bboxes,target_means,target_stds):
        """Change from cross offset to cross coordinate."""

        gt_x1 = gt_bboxes[:, 0]
        gt_y1 = gt_bboxes[:, 1]
        gt_x2 = gt_bboxes[:, 2]
        gt_y2 = gt_bboxes[:, 3]
        gt_xc = (gt_x1+gt_x2)*0.5
        gt_yc = (gt_y1+gt_y2)*0.5
        gt_w = gt_x2 - gt_x1
        gt_h = gt_y2 - gt_y1
        pre_xc_row = predict_cross[:, 0]
        pre_yc_row = predict_cross[:, 1]
        pre_w = predict_cross[:, 2]
        pre_xc_col = predict_cross[:, 3]
        pre_yc_col = predict_cross[:, 4]
        pre_h = predict_cross[:, 5]
        delta_x = (gt_xc - pre_xc_row) / pre_w
        delta_y = (gt_yc - pre_yc_col) / pre_h
        delta_w = torch.log(gt_w/pre_w)
        delta_h = torch.log(gt_h/pre_h)
        delta_bboxes = torch.stack([delta_x,delta_y,delta_w,delta_h],dim=-1)
        means = delta_bboxes.new_tensor(target_means)
        stds = delta_bboxes.new_tensor(target_stds)
        delta_bboxes = delta_bboxes.sub_(means).div_(stds)

        return delta_bboxes
    def cross2bbox(self,bboxes,cross,img_metas,mode='x1y1x2y2toxywxyh',clip=True):
        if mode =='x1y1x2y2toxywxyh':
            cross_x_w=(bboxes[...,0]+bboxes[...,2])*0.5
            cross_y_w=(bboxes[...,1]+bboxes[...,3])*0.5
            cross_x_h=cross_x_w
            cross_y_h=cross_y_w
            cross_w=bboxes[...,2]-bboxes[...,0]
            cross_h=bboxes[...,3]-bboxes[...,1]
            return torch.stack([cross_x_w,cross_y_w,cross_w,cross_x_h,cross_y_h,cross_h],dim=-1)
        if mode == 'xywxyhtox1y1x2y2':
            x1=cross[...,0]-cross[...,2]*0.5
            y1=cross[...,4]-cross[...,5]*0.5
            x2=cross[...,0]+cross[...,2]*0.5
            y2 = cross[..., 4] + cross[..., 5] * 0.5
            y_row=cross[...,1]
            x_col = cross[..., 3]
            max_shape=img_metas['img_shape']
            if max_shape is not None and clip:
                x1 = x1.clamp(min=0, max=max_shape[1])
                y1 = y1.clamp(min=0, max=max_shape[0])
                x2 = x2.clamp(min=0, max=max_shape[1])
                y2 = y2.clamp(min=0, max=max_shape[0])
            bbox=torch.stack([x1,y1,x2,y2],dim=-1)
            return bbox
        if mode == 'x1x2yxy1y2tox1y1x2y2':
            x1=cross[...,0]
            y1=cross[...,4]
            x2=cross[...,1]
            y2=cross[...,5]
            max_shape=img_metas['img_shape']
            if max_shape is not None and clip:
                x1 = x1.clamp(min=0, max=max_shape[1])
                y1 = y1.clamp(min=0, max=max_shape[0])
                x2 = x2.clamp(min=0, max=max_shape[1])
                y2 = y2.clamp(min=0, max=max_shape[0])
            bbox=torch.stack([x1,y1,x2,y2],dim=-1)
            return bbox

