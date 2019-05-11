# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

# TODO: write description

from torch import nn

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.
    """
    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        inference pipeline.
        Args:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): GT boxes in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor])
                During testing, it returns list[BoxList] containing scores & labels
        """
        # backbone network that extracts a feature map from an input image
        # -> ..backbone - build_backbone (ResNet50StagesTo4)
        features = self.backbone(images.tensors)

        # region proposal network that obtains object proposal boxes
        # ..rpn.rpn - build_rpn (RPNModule)
        proposals = self.rpn(images, features, targets)

        # ROI heads where each object is cropped and
        # its fine location and class (labels) are inferred
        if self.roi_heads: # Faster R-CNN
            # ..roi_heads.roi_heads - build_roi_heads (ROIBoxHead)
            x, result = self.roi_heads(features, proposals, targets)
        else: # RPN-only mode. Use the result of RPN
            x, result = features, proposals

        return result