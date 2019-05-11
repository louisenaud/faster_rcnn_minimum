# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# TODO: write description

import torch

from .box_head.box_head import build_roi_box_head

class CombinedROIHeads(torch.nn.ModuleDict):
    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()

    def forward(self, features, proposals, targets=None):
        x, detections = self.box(features, proposals, targets)
        return x, detections

def build_roi_heads(cfg, in_channels):
    roi_heads = []
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads