class Configuration():
    def __init__(self):
        self.PROJECT_NAME = 'gcn clustering'
        self.LOG_DIR = "./log/" #log dir and saved model dir
        self.DATA_DIR = "/xxx/"
        self.DEVICE_ID = "5"
        #data loader
        self.DATALOADER_NUM_WORKERS = 8
        self.BATCHSIZE = 128

        self.TRAIN_FEATS_PATH = '/xxx/datasets/gcn_cluster/train_feats.npy'
        self.TRAIN_KNN_DISTMAT_PATH = '/xxx/datasets/gcn_cluster/train_knn.npy'
        self.TRAIN_LABELS_PATH = '/xxx/datasets/gcn_cluster/train_labels.npy'

        self.TEST_FEATS_PATH = './log/feats.npy'
        self.TEST_KNN_DISTMAT_PATH = './log/knn.npy'
        self.TEST_LABELS_PATH = './log/labels.npy'

        self.SEED = 1
        self.NUM_HOP = [32,5]#[50, 5]
        self.NUM_ACTIVE_CONNECTION = 5

        #model
        self.INPUT_DIM = 2048
        self.MODEL_NAME = "gcn_duke"

        #loss
        self.LOSS_TYPE = 'softmax'
        self.LABELSMOOTH = 'off'

        #test
        self.TEST_WEIGHT = './log/gcn_duke_20.pth'    #gcn_20
        self.TEST_BATCHSIZE = 1


        #solver
        self.OPTIMIZER = 'Adam'
        self.BASE_LR = 0.01
        self.MOMENTUM = 0.9
        self.WEIGHT_DECAY = 0.0005
        self.BIAS_LR_FACTOR = 2
        self.WEIGHT_DECAY_BIAS = 0.0

        self.SOLVER_STEPS = [4,6,8,10,12,14,16,18]
        self.LR_DECAY_FACTOR = 0.6
        self.SOLVER_WARMUP_FACTOR = 0.5
        self.SOLVER_WARMUP_EPOCHS = 2
        self.SOLVER_WARMUP_METHOD = 'linear'

        self.LOG_PERIOD = 100 #iteration of display training log
        self.CHECKPOINT_PERIOD = 2 #save model period
        self.EVAL_PERIOD = self.CHECKPOINT_PERIOD
        self.MAX_EPOCHS = 20
