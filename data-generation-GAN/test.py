import os
import sys
from config.cfg import Cfg
import torch
from torch.backends import cudnn

sys.path.append('.')
from datasets import make_dataloader
from processor import do_inference
from model import make_model
from utils.logger import setup_logger

if __name__ == "__main__":
    Cfg.freeze()
    log_dir = Cfg.DATALOADER.LOG_DIR
    logger = setup_logger('pose-transfer-gan.test', log_dir)
    logger.info("Running with config:\n{}".format(Cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = Cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader = make_dataloader(Cfg)
    model_G, _, _, _ = make_model(Cfg)
    model_G.load_state_dict(torch.load(Cfg.TEST.WEIGHT))

    do_inference(Cfg, model_G, val_loader)