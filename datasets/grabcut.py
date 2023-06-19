import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2

class GrabCutdataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.image_dir = os.path.join(self.root, 'data_GT')
        self.mask_dir = os.path.join(self.root, 'boundary_GT')
        self.transform = transform
        self.images = os.listdir(self.image_dir)

    def __getitem__(self, idx):
        img, gt_mask = self._make_img_gt_point_pair(idx)

        sample = {'image': img, 'instances_mask': gt_mask}

        sample['meta'] = {'img_dir': self.img_path,
                          'gt_dir': self.mask_dir,
                          'im_size': (img.shape[0], img.shape[1]),
                          'img_name': self.img_name}

        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def __len__(self):
        return len(self.images)
    
    def _make_img_gt_point_pair(self, index):
        self.img_name = self.images[index]
        self.img_path = os.path.join(self.image_dir, self.images[index])
        self.mask_path = os.path.join(self.mask_dir, self.images[index][:-4]+'.bmp')

        # Read image
        _img = np.array(Image.open(self.img_path).convert('RGB'))

        # Read gt mask
        _gt_mask = (np.array(Image.open(self.mask_path)).astype(np.int32))
        _gt_mask[_gt_mask == 128] = -1
        _gt_mask[_gt_mask > 128] = 1

        return _img, _gt_mask
