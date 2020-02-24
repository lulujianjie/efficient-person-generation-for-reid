from .NewDataset import NewDataset


import torch
import torch.utils.data as data
import torchvision.transforms as T

def train_collate_fn(batch):
    imgs, imgpaths = zip(*batch)
    return torch.stack(imgs, dim=0),imgpaths

def make_dataloader(Cfg):
    transform = T.Compose([
        T.Resize(Cfg.MODEL.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    num_workers = Cfg.DATALOADER.DATALOADER_NUM_WORKERS
    dataset = NewDataset(Cfg, transform=transform)
    train_loader = data.DataLoader(
        dataset,
        batch_size=Cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=num_workers,
        sampler=None,
        collate_fn=train_collate_fn,  # customized batch sampler
        drop_last=False
    )
    return train_loader