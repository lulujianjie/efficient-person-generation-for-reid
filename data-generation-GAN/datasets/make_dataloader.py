import torchvision.transforms as T
from torch.utils.data import DataLoader

from .Market1501Pose import Market1501Pose
from .bases import ImageDataset

from config.cfg import Cfg


def make_dataloader(Cfg):
    train_transforms = T.Compose([
        T.Resize(Cfg.MODEL.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_transforms = T.Compose([
        T.Resize(Cfg.MODEL.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_set = ImageDataset(
        Market1501Pose(data_dir=Cfg.DATALOADER.DATA_DIR, verbose=True, split='train', restore=True),
        transform=train_transforms,
        epoch_size='medium'
    )
    test_set = ImageDataset(
        Market1501Pose(data_dir=Cfg.DATALOADER.DATA_DIR, verbose=True, split='test', restore=True),
        transform=test_transforms,
        epoch_size='large'
    )

    train_loader = DataLoader(train_set,
        batch_size=Cfg.SOLVER.BATCHSIZE,
        shuffle=True,
        num_workers=Cfg.DATALOADER.DATALOADER_NUM_WORKERS,
        sampler = None,
        drop_last = True
    )

    test_loader = DataLoader(test_set,
        batch_size=Cfg.TEST.BATCHSIZE,
        shuffle=False,
        num_workers=Cfg.DATALOADER.DATALOADER_NUM_WORKERS,
        drop_last = False
    )
    return train_loader, test_loader

if __name__ == '__main__':
    #remove . for bases and Market1501Pose
    train_loader, _ = make_dataloader(Cfg)
    for idx, data_dict in enumerate(train_loader):
        print(data_dict['img1'].shape)
        print(data_dict['pose1'].shape)
        print(data_dict['img2'].shape)
        print(data_dict['pose2'].shape)