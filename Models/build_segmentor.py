
import torch.nn as nn
from Models.SCAUNet import SCAUNet

__all__ = ['Model', 'SCAUNet']


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        backbone_name = cfg.model['backbone']['type'] if cfg.model['backbone']['type'] else None
        decode_name = cfg.model['decode_head']['type']
        backbone_class = globals()[backbone_name] if backbone_name else None
        decode_class = globals()[decode_name]
        self.backbone = backbone_class(**cfg.model['backbone']) if backbone_name else None
        self.decode_head = decode_class(**cfg.model['decode_head'])

    def forward(self, x):
        if self.backbone:
            x = self.backbone(x)
        out = self.decode_head(x)
        return out
