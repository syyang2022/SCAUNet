
from mmcv import Config
from Models.build_segmentor import Model


def build_model(cfg):
    model = Model(cfg)
    return model


if __name__ == '__main__':
    config = '/data1/ppw/works/All_ISTD/configs/segnext/segnext_tiny_512x512_800e_nuaa.py'
    cfg = Config.fromfile(config)
    model = build_model(cfg)
    total = sum([param.nelement() for param in model.parameters()])
    print(total)
    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(8, 3, 512, 512)
    model = model.to(device)
    x = x.to(device)
    out = model(x)
    print(out.size())
