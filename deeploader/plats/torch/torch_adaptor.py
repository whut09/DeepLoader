import os
import random

import numpy as np
from torch.utils import data
from PIL import Image

class TorchAdaptor(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, dataset, transform, for_training=True):
        """Initialize and preprocess the CelebA dataset."""
        self.transform = transform
        self.for_training = for_training
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset.getData(index)
        image = item['img']
        label = item['label']
        if 'torch' in str(type(self.transform)):
            image = Image.fromarray(image, mode="RGB")
        # return self.transform(image), np.array(label).astype(np.int64)  # torch.FloatTensor(label)
        return self.transform(image), np.array(label).astype(np.int64), np.array(index).astype(np.int64)

    def __len__(self):
        """Return the number of images."""
        return self.dataset.size()

    def size(self):
        return self.dataset.size()