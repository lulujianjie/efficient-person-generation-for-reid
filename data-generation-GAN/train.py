import os

from config.cfg import Cfg
from torch.backends import cudnn

from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer, WarmupMultiStepLR
from loss import make_loss

from processor import do_train


if __name__ == '__main__':

    Cfg.freeze()
    log_dir = Cfg.DATALOADER.LOG_DIR
    logger = setup_logger('pose-transfer-gan.train', log_dir)
    logger.info("Running with config:\n{}".format(Cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = Cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True
    # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.

    train_loader, val_loader = make_dataloader(Cfg)
    model_G, model_Dip, model_Dii, model_D_reid = make_model(Cfg)

    optimizerG = make_optimizer(Cfg, model_G)
    optimizerDip = make_optimizer(Cfg, model_Dip)
    optimizerDii = make_optimizer(Cfg, model_Dii)

    schedulerG = WarmupMultiStepLR(optimizerG, Cfg.SOLVER.STEPS, Cfg.SOLVER.GAMMA,
                                  Cfg.SOLVER.WARMUP_FACTOR,
                                  Cfg.SOLVER.WARMUP_EPOCHS, Cfg.SOLVER.WARMUP_METHOD)
    schedulerDip = WarmupMultiStepLR(optimizerDip, Cfg.SOLVER.STEPS, Cfg.SOLVER.GAMMA,
                                  Cfg.SOLVER.WARMUP_FACTOR,
                                  Cfg.SOLVER.WARMUP_EPOCHS, Cfg.SOLVER.WARMUP_METHOD)
    schedulerDii = WarmupMultiStepLR(optimizerDii, Cfg.SOLVER.STEPS, Cfg.SOLVER.GAMMA,
                                  Cfg.SOLVER.WARMUP_FACTOR,
                                  Cfg.SOLVER.WARMUP_EPOCHS, Cfg.SOLVER.WARMUP_METHOD)
    GAN_loss, L1_loss, ReID_loss = make_loss(Cfg)
    do_train(
            Cfg,
            model_G, model_Dip, model_Dii, model_D_reid,
            train_loader, val_loader,
            optimizerG, optimizerDip, optimizerDii,
            GAN_loss, L1_loss, ReID_loss,
            schedulerG, schedulerDip, schedulerDii
        )
