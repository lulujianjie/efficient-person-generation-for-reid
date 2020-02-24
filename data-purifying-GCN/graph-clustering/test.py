import os
import sys
from config.config import Configuration
import torch
from torch.backends import cudnn

sys.path.append('.')
from datasets import make_dataloader
from processor import do_inference
from model import make_model
from utils.logger import setup_logger

if __name__ == "__main__":
    Cfg = Configuration()
    log_dir = Cfg.LOG_DIR
    logger = setup_logger('{}.test'.format(Cfg.PROJECT_NAME), log_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = Cfg.DEVICE_ID
    cudnn.benchmark = True
    # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.

    train_loader, test_loader = make_dataloader(Cfg)
    model = make_model(Cfg)
    model.load_state_dict(torch.load(Cfg.TEST_WEIGHT))

    do_inference(Cfg, model, test_loader)