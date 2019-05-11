# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from torch import nn
from faster_rcnn_minimum.layers import ROIAlign
from .utils import cat

# TODO: write description

class Pooler(nn.Module):
    def __init__(self, output_size, scales, sampling_ratio):
        super(Pooler, self).__init__()
        poolers = [ROIAlign(output_size, spatial_scale=scales[0], sampling_ratio=sampling_ratio)]
        self.poolers = nn.ModuleList(poolers)

    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, boxes):
        rois = self.convert_to_roi_format(boxes)
        return self.poolers[0](x[0], rois)

