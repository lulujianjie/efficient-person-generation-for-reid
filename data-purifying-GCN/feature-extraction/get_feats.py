import os
import sys
from config.cfg import Cfg

from torch.backends import cudnn

sys.path.append('.')
from datasets import make_dataloader
from processor import do_inference
from model import make_model

from utils.logger import setup_logger

if __name__ == "__main__":
    # with open('cfg_test.json') as f:
    #     cfg = json.load(f)
    Cfg.freeze()
    log_dir = Cfg.DATALOADER.LOG_DIR
    logger = setup_logger('Extract Feats', log_dir)
    logger.info("Running with config:\n{}".format(Cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = Cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    val_loader = make_dataloader(Cfg)
    model = make_model(Cfg,255)
    model.load_param(Cfg.TEST.WEIGHT)

    do_inference(Cfg, model, val_loader)