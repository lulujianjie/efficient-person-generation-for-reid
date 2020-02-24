import os

import torch
from config.config import Configuration
from torch.backends import cudnn

from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer, WarmupMultiStepLR
from loss import make_loss

from processor import do_train


if __name__ == '__main__':

    Cfg = Configuration()
    log_dir = Cfg.LOG_DIR
    logger = setup_logger('{}'.format(Cfg.PROJECT_NAME), log_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = Cfg.DEVICE_ID
    cudnn.benchmark = True
    # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.

    train_loader, test_loader = make_dataloader(Cfg)
    model = make_model(Cfg)


    optimizer = make_optimizer(Cfg, model)
    scheduler = WarmupMultiStepLR(optimizer, Cfg.SOLVER_STEPS, Cfg.LR_DECAY_FACTOR,
                                  Cfg.SOLVER_WARMUP_FACTOR,
                                  Cfg.SOLVER_WARMUP_EPOCHS, Cfg.SOLVER_WARMUP_METHOD)
    loss_func = make_loss(Cfg, num_classes=2)
    do_train(Cfg, model, train_loader, test_loader, optimizer,
            scheduler,  # modify for using self trained model
            loss_func)
