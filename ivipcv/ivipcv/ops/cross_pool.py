import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
# from torch.nn.modules.utils import _pair

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['cross_pool_forward', 'cross_pool_backward'])


class CrossPoolFunction(Function):

    @staticmethod
    def symbolic(g, input, rois, spatial_scale, pool_mode):
        return g.op(
            'IVIPCrossPool',
            input,
            rois,
            spatial_scale=spatial_scale,
            pool_mode = pool_mode)

    @staticmethod
    def forward(ctx, input, rois, spatial_scale=1.0, pool_mode=0, axis=0):
        ctx.spatial_scale = spatial_scale
        # 内部使用0代表max，1代表average
        ctx.pool_mode = 0
        if pool_mode == "avg":
            ctx.pool_mode = 1
        ctx.axis = int(axis)
        ctx.input_shape = input.size()

        # 检查ROI尺寸      
        assert rois.size(-1) == 6, 'RoI must be (x1, x2, y, x, y1, y2).'

        # 根据axis参数确定在输出通道总数上的倍率
        axis_ratio = max(axis, 1)

        # 计算输出通道数
        output_channel = input.size(1) * axis_ratio
        assert output_channel != 0, "output_channel cann't be zero."

        # output_shape = input.size() # output_shape same as input.size(). eg.(1,1,7,7)

        # 每条线生成的特征谱单独拿出来，在channel维度上扩充
        output_shape = [input.size(0), output_channel, input.size(2), input.size(3)]
        output = input.new_zeros(output_shape)
        argmax = input.new_zeros(output_shape) # argmax = input.new_zeros(output_shape, dtype=torch.int)
        argmax_y = input.new_zeros(output_shape)
        ext_module.cross_pool_forward(
            input,
            rois,
            output,
            argmax,
            argmax_y,
            spatial_scale = ctx.spatial_scale,
            pool_mode = ctx.pool_mode,
            axis = ctx.axis)

        ctx.save_for_backward(rois, argmax, argmax_y)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, argmax, argmax_y = ctx.saved_tensors
        grad_input = grad_output.new_zeros(ctx.input_shape)
        ext_module.cross_pool_backward(
            grad_output,
            rois,
            argmax,
            argmax_y,
            grad_input,
            spatial_scale=ctx.spatial_scale,
            pool_mode = ctx.pool_mode,
            axis = ctx.axis)

        return grad_input, None, None, None, None # Return grad_input


cross_pool = CrossPoolFunction.apply


class CrossPool(nn.Module):

    def __init__(self, spatial_scale=1.0, pool_mode = "max", axis = "HW"):
        super(CrossPool, self).__init__()
        self.spatial_scale = float(spatial_scale)

        pool_mode = pool_mode.lower()
        assert pool_mode == "max" or pool_mode == "avg", "CrossPool parameter pool_mode must be \"max\" or \"avg\""
        self.pool_mode = pool_mode

        # 对axis参数进行转换
        axis = axis.upper()
        if axis == "HW" or axis == "WH":
            self.axis = 2  # 横线和纵线都计算
        elif axis == "W":
            self.axis = 0  # 只计算W方向线（横线）
        elif axis == "H":
            self.axis = 1  # 只计算H方向线（纵线）
        else:
            raise Exception("axis must be W or H or WH, get {}".format(axis))

    def forward(self, input, rois):
        return cross_pool(input, rois, self.spatial_scale, self.pool_mode, self.axis)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'spatial_scale={self.spatial_scale})'
        s += f'pool_mode={self.pool_mode})'
        s += f'axis={self.axis})'
        return s
