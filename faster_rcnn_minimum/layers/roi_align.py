# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from faster_rcnn_minimum import _C


class _ROIAlign(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        output = _C.roi_align_forward(
            input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio
        )
        return output


roi_align = _ROIAlign.apply

class ROIAlign(nn.Module):
    """
    ROIAlign layer that crops the feature map at the sub-pixel level.
    """
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        """
        crop the feature map with ROIs
        Args:
            input (torch.Tensor): input feature map with a dimension of [B, C, H, W]
            rois (torch.Tensor): bbox tensor with a dimension of [K, 5]
                                 where K is _C.MODEL.RPN.POST_NMS_TOP_N_TEST
        Returns:
            feature map (torch.Tensor): cropped feature map with a dimension of [K, C, R, R]
                                 where K is _C.MODEL.RPN.POST_NMS_TOP_N_TEST and
                                 R is defined as _C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        """
        return roi_align(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
        )
