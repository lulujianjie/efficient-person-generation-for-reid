import torch

def make_optimizer(Cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = Cfg.SOLVER.BASE_LR
        weight_decay = Cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = Cfg.SOLVER.BASE_LR * Cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = Cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "betas": (0.5, 0.999), "weight_decay": weight_decay}]

    optimizer = getattr(torch.optim, Cfg.SOLVER.OPTIMIZER)(params)
    return optimizer