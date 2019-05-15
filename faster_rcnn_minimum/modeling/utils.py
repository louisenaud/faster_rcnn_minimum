# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# TODO: write description

import torch

def cat(tensors, dim=0):
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)