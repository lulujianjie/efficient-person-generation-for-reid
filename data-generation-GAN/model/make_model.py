import torch
import torch.nn as nn
from torch.nn import init
from .backbones.basicblock import ImageEncoder, PoseEncoder, PATNs, ImageGenerator, ResBlock
from .backbones.reid_D import ReidDiscriminator

class PATNetwork(nn.Module):
    def __init__(self, cfg):
        super(PATNetwork, self).__init__()
        self.image_encoder = ImageEncoder(nlayers=cfg.MODEL.NUM_LAYERS_IENCODER)
        self.pose_encoder = PoseEncoder(nlayers=cfg.MODEL.NUM_LAYERS_PENCODER)
        self.PATNs = PATNs(inplanes=256, nblocks=cfg.MODEL.NUM_BLOCKS_PATN)
        self.image_generator = ImageGenerator(nlayers=cfg.MODEL.NUM_LAYERS_IGENERATOR)
    def forward(self, input):
        img1, pose2 = input
        fimg = self.image_encoder(img1)
        fpose = self.pose_encoder(pose2)

        fimg = self.PATNs(input=(fimg, fpose))

        output = self.image_generator(fimg)
        return output

class ResNet(nn.Module):
    def __init__(self, dim, nblocks):
        super(ResNet, self).__init__()
        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(dim, 64, kernel_size=7, stride=1, padding=0),
                  nn.BatchNorm2d(64),
                  nn.ReLU(True),
                  nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                  nn.BatchNorm2d(128),
                  nn.ReLU(True),
                  nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                  nn.BatchNorm2d(256),
                  nn.ReLU(True)]

        for i in range(nblocks):
            layers.append(ResBlock(256))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def make_model(cfg):
    model_G = PATNetwork(cfg)
    model_D_ip = ResNet(3+6, cfg.MODEL.NUM_BLOCKS_RESNET)
    model_D_ii = ResNet(3+3, cfg.MODEL.NUM_BLOCKS_RESNET)
    model_D_reid = ReidDiscriminator(cfg)
    print('=>Initializing model...')
    init_weights(model_G)
    init_weights(model_D_ip)
    init_weights(model_D_ii)
    model_D_reid.load_param(cfg.MODEL.REID_WEIGHT)
    return model_G, model_D_ip, model_D_ii, model_D_reid

if __name__ == '__main__':
    from config.cfg import Cfg

    Cfg.freeze()
    model_G, _,_ = make_model(Cfg)
    model_G.to('cuda')
    input1 = torch.randn((1, 3, 128, 64)).to('cuda')
    input2 = torch.randn((1, 12, 128, 64)).to('cuda')
    output = model_G(input=(input1,input2))
    #output_D = model_D(output)
    print(output.shape)
    #print(output_D.shape)