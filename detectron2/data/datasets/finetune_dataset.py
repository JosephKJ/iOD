import os
import torch
import torch.utils.data as data
from fvcore.common.file_io import PathManager


class ImageStoreDataset(data.Dataset):
    def __init__(self, cfg):
        path = os.path.join(cfg.WG.IMAGE_STORE_LOC, 'image_store.pth')
        with PathManager.open(path, 'rb') as f:
            image_store = torch.load(f)
        self.images = image_store.retrieve()

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.images)
