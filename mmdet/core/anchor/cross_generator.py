import torch

from .builder import ANCHOR_GENERATORS


@ANCHOR_GENERATORS.register_module()
class CrossGenerator(object):

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_cross(self, featmap_size, stride=16, device='cuda'):
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0., feat_w, device=device) * stride
        shift_y = torch.arange(0., feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        stride = shift_x.new_full((shift_xx.shape[0], ), stride)
        base_scale=8
        x1_row=shift_xx-base_scale*stride*0.5
        x2_row=shift_xx+base_scale*stride*0.5
        yc_row=shift_yy
        y1_col = shift_yy - base_scale * stride * 0.5
        y2_col = shift_yy + base_scale * stride * 0.5
        xc_col = shift_xx
        shifts = torch.stack([x1_row, x2_row, yc_row, xc_col, y1_col, y2_col], dim=-1)
        shifts=shifts.view(feat_h,feat_w,6)
        all_cross = shifts.to(device)
        return all_cross

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid
