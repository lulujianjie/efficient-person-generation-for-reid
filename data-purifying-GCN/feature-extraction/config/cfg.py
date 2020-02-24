from yacs.config import CfgNode as cfg
#config tree
Cfg = cfg()

Cfg.DATALOADER = cfg()
Cfg.DATALOADER.LOG_DIR = "./log/" #log dir and saved model dir
Cfg.DATALOADER.DATALOADER_NUM_WORKERS = 8


Cfg.MODEL = cfg()
Cfg.MODEL.INPUT_SIZE = [256,128]#[256, 128] #HxW
Cfg.MODEL.MODEL_NAME = "resnet50"
Cfg.MODEL.DEVICE_ID = "5"#
Cfg.MODEL.LAST_STRIDE = 1
Cfg.MODEL.MODEL_NECK = 'bnneck'#'bnneck'
Cfg.MODEL.NECK_FEAT = "after"#after

Cfg.TEST = cfg()
Cfg.TEST.IMS_PER_BATCH = 128
Cfg.TEST.FEAT_NORM = "yes"#yes
Cfg.TEST.WEIGHT = '/xxx/resnet50_person_reid_gcn.pth'
Cfg.TEST.DIST_MAT = Cfg.DATALOADER.LOG_DIR+"dist_mat.npy"
Cfg.TEST.IMG_PATH = Cfg.DATALOADER.LOG_DIR+"img_path.npy"
Cfg.TEST.FEATS = Cfg.DATALOADER.LOG_DIR+"feats.pth"

Cfg.TEST.FIRST_QUERY = 0
Cfg.TEST.NUM_QUERY = 100
Cfg.TEST.DIST_METHOD = 'cosine'#'euclidean'#'cosine'