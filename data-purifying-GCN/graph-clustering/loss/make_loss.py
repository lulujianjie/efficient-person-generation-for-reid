import torch.nn.functional as F

from .softmax_loss import CrossEntropyLabelSmooth



def make_loss(Cfg, num_classes):
    if Cfg.LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    def loss_func(score, target):
        if Cfg.LOSS_TYPE == 'softmax':
            # print('Train with center loss, the loss type is triplet+center_loss')
            if Cfg.LABELSMOOTH == 'on':
                return xent(score, target)
            else:
                return F.cross_entropy(score, target)
        else:
            print('unexpected loss type')
    return loss_func