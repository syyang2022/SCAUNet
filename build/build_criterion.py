
from utils.loss import *
from torch import autograd
__all__ = ['build_criterion', 'SoftIoULoss', 'BCEWithLogits', 'CrossEntropy', 'Multi_Loss']


#  TODO Multiple loss functions
def build_criterion(cfg):
    criterion_name = cfg.model['loss']['type']
    criterion_class = globals()[criterion_name]
    criterion = criterion_class(**cfg.model['loss'])
    return criterion




