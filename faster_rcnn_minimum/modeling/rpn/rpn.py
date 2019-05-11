# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torch.nn.functional as F
from torch import nn

from faster_rcnn_minimum.modeling import registry
from faster_rcnn_minimum.modeling.box_coder import BoxCoder
from .anchor_generator import make_anchor_generator
from .inference import make_rpn_postprocessor

# TODO: write description

@registry.RPN_HEADS.register("SingleConvRPNHead")
class RPNHead(nn.Module):
    def __init__(self, cfg, in_channels, num_anchors):
        """
        Args:
            cfg: config
            in_channels (int): number of input feature channels
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        feature = x[0]
        t = F.relu(self.conv(feature))
        logits.append(self.cls_logits(t))
        bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg



class RPNModule(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(RPNModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator(cfg)

        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
        head = rpn_head(
            cfg, in_channels, anchor_generator.num_anchors_per_location()[0]
        )

        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_test = box_selector_test

    def forward(self, images, features, targets=None):
        """
        Args:
            images (ImageList):
            features (list[Tensor]):
            targets (list[BoxList]):

        Returns:

        """
        objectness, rpn_box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)

        if self.training:
            raise NotImplementedError("training not implemented")
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes


def build_rpn(cfg, in_channels):
    return RPNModule(cfg, in_channels)