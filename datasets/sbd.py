import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import random
from torchvision import transforms

import json
import scipy
import cv2


class SBDdataset(Dataset):
    def __init__(self, root, image_set, transform=None, preprocess=False):

        if image_set == 'train':
            print('Loading SBD Dataset for training...')
        elif image_set == 'val':
            print('Loading SBD Dataset for validation...')

        self.root = root
        self.img_dir = os.path.join(self.root, 'img')
        self.mask_dir = os.path.join(self.root, 'inst')
        self.image_set = image_set
        self.transform = transform
        self.obj_list_file = os.path.join(self.root, self.image_set + '_instances.txt')

        split_f = os.path.join(self.root, self.image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as fh:
            file_names = [x.strip() for x in fh.readlines()]

        self.im_ids = file_names
        self.images = [os.path.join(self.img_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(self.mask_dir, x + ".mat") for x in file_names]
        assert (len(self.images) == len(self.masks))

        # Precompute the list of objects and their categories for each image
        if (not self._check_preprocess()) or preprocess:
            self._preprocess()

        # Build the list of objects
        self.obj_list = []
        num_images = 0
        for ii in range(len(self.im_ids)):
            if self.im_ids[ii] in self.obj_dict.keys():
                flag = False
                for jj in range(len(self.obj_dict[self.im_ids[ii]])):
                    if self.obj_dict[self.im_ids[ii]][jj] != -1:
                        self.obj_list.append([ii, jj])
                        flag = True
                if flag:
                    num_images += 1

        # Display stats
        print('\tNumber of images: {:d}\n\tNumber of objects: {:d}\n'.format(num_images, len(self.obj_list)))

    def __getitem__(self, idx):
        img, gt_mask = self._make_img_gt_point_pair(idx)

        sample = {'image': img, 'gt_mask': gt_mask}
        _im_ii = self.obj_list[idx][0]
        _obj_ii = self.obj_list[idx][1]
        sample['meta'] = {'image': str(self.im_ids[_im_ii]),
                          'object': str(_obj_ii),
                          'im_size': (img.shape[0], img.shape[1]),
                          'category': self.obj_dict[self.im_ids[_im_ii]][_obj_ii]}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.obj_list)

    def _make_img_gt_point_pair(self, index):
        _im_idx = self.obj_list[index][0]
        _obj_idx = self.obj_list[index][1]

        # Read image
        _img = np.array(Image.open(self.images[_im_idx]).convert('RGB')).astype(np.float32)

        # Read gt mask
        _tmp = scipy.io.loadmat(self.masks[_im_idx])["GTinst"][0]["Segmentation"][0]
        _gt_mask = (_tmp == (_obj_idx + 1)).astype(np.float32)

        if self.image_set == 'train':
            # Resize image & gt mask
            _img, _gt_mask = self.resize_shorter480(_img, _gt_mask)

            # Crop image & gt mask (400x400 patch around object instance)
            crop_h, crop_w = self.extancemask2crophw(_gt_mask)
            _img = _img[crop_h: crop_h + 400, crop_w: crop_w + 400]
            _gt_mask = _gt_mask[crop_h:crop_h + 400, crop_w:crop_w + 400]

        return _img, _gt_mask

    def _check_preprocess(self):
        # Check that the file with categories is there and with correct size
        _obj_list_file = self.obj_list_file
        if not os.path.isfile(_obj_list_file):
            return False
        else:
            self.obj_dict = json.load(open(_obj_list_file, 'r'))
            return list(np.sort([str(x) for x in self.obj_dict.keys()])) == list(np.sort(self.im_ids))

    def _preprocess(self):
        # Get all object instances and their category
        self.obj_dict = {}
        obj_counter = 0
        for i in range(len(self.im_ids)):
            # Read object masks and get number of objects
            tmp = scipy.io.loadmat(self.masks[i])
            _mask = tmp["GTinst"][0]["Segmentation"][0]
            _cat_ids = tmp["GTinst"][0]["Categories"][0].astype(int)

            _mask_ids = np.unique(_mask)
            n_obj = _mask_ids[-1]
            assert (n_obj == len(_cat_ids))

            for j in range(n_obj):
                temp = np.where(_mask == j + 1)
                # obj_area = len(temp[0])
                # if obj_area < self.area_thres:
                #     _cat_ids[j] = -1
                obj_counter += 1

            self.obj_dict[self.im_ids[i]] = np.squeeze(_cat_ids, 1).tolist()  # np.squeeze: axis=1

        # Save it to file for future reference
        with open(self.obj_list_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(self.im_ids[0], json.dumps(self.obj_dict[self.im_ids[0]])))
            for i in range(1, len(self.im_ids)):
                outfile.write(',\n\t"{:s}": {:s}'.format(self.im_ids[i], json.dumps(self.obj_dict[self.im_ids[i]])))
            outfile.write('\n}\n')

        print('Pre-processing finished')

    def resize_shorter480(self, img, gt_mask):
        ori_h, ori_w = img.shape[0], img.shape[1]
        if ori_w >= ori_h:
            if ori_h == 480:
                return img, gt_mask
            new_h = 480
            new_w = int((ori_w / ori_h) * 480)
        else:
            if ori_w == 480:
                return img, gt_mask
            new_w = 480
            new_h = int((ori_h / ori_w) * 480)

        output_size = (new_w, new_h)

        new_img = cv2.resize(img, output_size, interpolation=cv2.INTER_CUBIC)
        new_gt_mask = cv2.resize(gt_mask, output_size, interpolation=cv2.INTER_NEAREST)

        return new_img, new_gt_mask

    def extancemask2crophw(self, gt):

        existance_mask = gt > 0.5
        max_h, max_w = existance_mask.shape[0]-400, existance_mask.shape[1]-400     #cropped size: (400,400)

        if existance_mask.max()==0:
            crop_h, crop_w = random.randint(0, max_h), random.randint(0, max_w)
        else:
            scrpoints = np.where(existance_mask!=0)
            idx = random.randint(0, len(scrpoints[0])-1)
            include_point = (scrpoints[0][idx], scrpoints[1][idx])

            min_h, min_w = max(0, include_point[0]-400), max(0, include_point[1]-400)
            max_h, max_w = min(max_h, include_point[0]), min(max_w, include_point[1])

            try: crop_h, crop_w = random.randint(min_h, max_h), random.randint(min_w, max_w)
            except: raise ValueError

        return crop_h, crop_w

