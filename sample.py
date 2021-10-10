import torch
from ivipcv.ops import StarPool

feat = torch.randn(3, 3, 9, 9, requires_grad=True).cuda()

a = torch.zeros((3, 9, 9, 3))
b = torch.zeros((3, 9, 9, 3))

a[:, :, :, 0] = 0  # x1
a[:, :, :, 1] = 8  # x2
a[:, :, :, 2] = 4  # y

b[:, :, :, 0] = 4  # x
b[:, :, :, 1] = 3  # y1
b[:, :, :, 2] = 8  # y2

rois = torch.cat((a, b), 3).cuda()

# pool_mode = "max" 十字的MaxPooling
layer = StarPool(spatial_scale=1.0, pool_mode="max")
output = layer(feat, rois) # output.size() 应为[3,6,9,9]
print(output.size())
