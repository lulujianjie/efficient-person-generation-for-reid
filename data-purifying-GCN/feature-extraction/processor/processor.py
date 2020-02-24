import torch
import torch.nn as nn
import numpy as np
import logging

from utils.metrics import Dist_Mat

def do_inference(Cfg, model, data_loader):
    device = "cuda"
    logger = logging.getLogger("Extract Feats")
    logger.info("Enter inferencing")

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        else:
            model.to(device)
    model.eval()

    evaluator = Dist_Mat(Cfg.TEST.FIRST_QUERY, Cfg.TEST.NUM_QUERY, Cfg.TEST.FEAT_NORM, method=Cfg.TEST.DIST_METHOD)
    img_path_list = []
    for idx, (img, imgpath) in enumerate(data_loader):
        if (idx+1) % 100 == 0:
            logger.info("Finished 12800 samples")
        with torch.no_grad():
            img_path_list.extend(imgpath)

            img = img.to(device) if torch.cuda.device_count() >= 1 else img
            feat = model(img)
            evaluator.update(feat)

    distmat,feats = evaluator.compute()
    logger.info("Finished inference")
    np.save(Cfg.TEST.DIST_MAT, distmat)
    np.save(Cfg.TEST.IMG_PATH, img_path_list)
    torch.save(feats, Cfg.TEST.FEATS)
