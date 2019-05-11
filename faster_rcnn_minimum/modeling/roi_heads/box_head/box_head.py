# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor

# TODO: write description

class ROIBoxHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)

    def forward(self, features, proposals, targets=None):
        """
        - crop the object ROI using ROIAlign
        - extract features from the cropped feature map
        - predict the class and box location using box predictor network

        Args:
            features:
            proposals:
            targets:
        Returns:
        """

        x = self.feature_extractor(features, proposals)

        class_logits, box_regression = self.predictor(x)

        if self.training:
            raise NotImplementedError("training not implemented")

        result = self.post_processor((class_logits, box_regression), proposals)
        return x, result


def build_roi_box_head(cfg, in_channels):
    return ROIBoxHead(cfg, in_channels)