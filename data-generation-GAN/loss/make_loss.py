import torch
import torch.nn as nn
from torch.autograd import Variable
from .L1perceptual import L1_plus_perceptualLoss

class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.real_label = 1.0
        self.fake_label = 0.0
        self.real_label_var = None
        self.fake_label_var = None
        self.loss = nn.BCELoss()

    def get_target_tensor(self, input, using_real_label):
        #target_tensor = None
        if using_real_label:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = torch.FloatTensor(input.size()).fill_(self.real_label).to('cuda')
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = torch.FloatTensor(input.size()).fill_(self.fake_label).to('cuda')
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, using_real_label):
        target_tensor = self.get_target_tensor(input, using_real_label)
        return self.loss(input, target_tensor)


def make_loss(cfg):
    if cfg.LOSS.L1_TYPE == 'L1+perL1':
        L1_loss = L1_plus_perceptualLoss(
            lambda_L1=cfg.LOSS.LAMBDA_L1,
            lambda_perceptual=cfg.LOSS.LAMBDA_PER,
            perceptual_layers=cfg.LOSS.NUM_LAYERS_VGG,
            percep_is_l1=1
        )
    elif cfg.LOSS.L1_TYPE == 'L1':
        L1_loss = cfg.LOSS.LAMBDA_L1*nn.L1Loss()
    GAN_Loss = GANLoss()
    ReID_Loss = nn.HingeEmbeddingLoss(margin=1, size_average=True, reduce=None, reduction='mean')
    return GAN_Loss, L1_loss, ReID_Loss