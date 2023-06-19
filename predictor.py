import torch
import numpy as np
import torchvision

from generate_click import generate_click
from BRS.InputCorrectionTorch import BRS


class BasePredictor(object):
    def __init__(self, net, device):
        self.net = net
        self.device = device
        self.original_image = None
        self.max_iact = float(255.0)

    def set_input_image(self, image_nd, instances_mask, image_name, save_res_dir):
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)

        self.gt_mask = instances_mask
        self.img_name = image_name
        self.img_size = instances_mask.shape
        self.save_dir = save_res_dir

        # Initialize clicks
        self.click_list = []

        # Initialize maps
        self.pclick_map = np.zeros(self.img_size, dtype=np.bool)
        self.nclick_map = np.zeros(self.img_size, dtype=np.bool)
        self.piactmap = np.ones(self.img_size, dtype=np.float32)
        self.niactmap = np.ones(self.img_size, dtype=np.float32)

    def get_prediction(self, pred_mask):
        # Generate click
        last_click = self._generate_click(pred_mask)

        # Get prediction
        prediction, iact = self._get_prediction(self.original_image, last_click)

        return prediction.cpu().numpy()[0, 0], iact.cpu().numpy()

    def _get_prediction(self, image_nd, last_click):
        # Generate interaction map
        iact_nd = self._generate_iact(last_click, self.img_size, self.max_iact)
        iact_torch = torch.tensor(1 - iact_nd, dtype=torch.float32, device=self.device)

        # Perform forward pass
        segmap = self.net(image_nd, iact_torch)

        return segmap, iact_torch

    def _generate_click(self, pred_mask):
        is_pos, yx_click = generate_click(self.gt_mask, pred_mask, self.pclick_map + self.nclick_map)

        # update clicks
        self.click_list.append([yx_click[0], yx_click[1], is_pos])
        if is_pos == 1:
            self.pclick_map[yx_click[0], yx_click[1]] = True
        else:
            self.nclick_map[yx_click[0], yx_click[1]] = True

        return (is_pos, yx_click)

    def _generate_iact(self, click, image_size, max_iact):
        is_pos, yx_click = click
        y_linspace = np.linspace(0, image_size[0] - 1, image_size[0], dtype=np.float32)
        x_linspace = np.linspace(0, image_size[1] - 1, image_size[1], dtype=np.float32)

        x_meshgrid, y_meshgrid = np.meshgrid(x_linspace, y_linspace)

        single_iactmap = np.sqrt(pow(y_meshgrid - yx_click[0], 2) + pow(x_meshgrid - yx_click[1], 2))
        single_iactmap = np.minimum(single_iactmap, max_iact) / max_iact

        if is_pos == 1:
            self.piactmap = np.minimum(self.piactmap, single_iactmap)
        else:
            self.niactmap = np.minimum(self.niactmap, single_iactmap)

        iact_nd = np.expand_dims(np.stack((self.piactmap, self.niactmap), axis=0), axis=0)

        return iact_nd


class BRSPredictor(BasePredictor):
    def __init__(self, model, device):
        super().__init__(model, device)
        self.InputCorrection = BRS(self.net, self.device)
        self.opt_data = None

        self.save_iacts = False

    def set_input_image(self, image_nd, instances_mask, image_name, save_res_dir):
        super().set_input_image(image_nd, instances_mask, image_name, save_res_dir)
        self.opt_data = None
        self.target_map = np.zeros([1, 1, self.img_size[0], self.img_size[1]], dtype=np.float32)
        self.valid_map = np.zeros([1, 1, self.img_size[0], self.img_size[1]], dtype=np.float32)

    def _get_clicks_maps_nd(self, last_click):
        is_pos, (y, x) = last_click
        if is_pos:
            self.target_map[0, 0, y, x] = 1
        else:
            self.target_map[0, 0, y, x] = 0
        self.valid_map[0, 0, y, x] = 1

        with torch.no_grad():
            target_map = torch.from_numpy(self.target_map).to(self.device)
            valid_map = torch.from_numpy(self.valid_map).to(self.device)

        return target_map, valid_map

    def _get_prediction(self, image_nd, last_click):
        target_map, valid_map = self._get_clicks_maps_nd(last_click)

        if self.opt_data is None:
            self.opt_data = torch.zeros((image_nd.shape[0], 2, image_nd.shape[2], image_nd.shape[3]),
                                        device=self.device, dtype=torch.float32)

        with torch.no_grad():
            # Generate interaction map
            iact_nd = self._generate_iact(last_click, self.img_size, self.max_iact)
            iact_torch = torch.tensor(1 - iact_nd, dtype=torch.float32, device=self.device)

            # Perform forward pass
            segmap = self.net(image_nd, iact_torch)
            if self.save_iacts:
                torchvision.utils.save_image(iact_torch[0,0], self.save_dir / f'{self.img_name}_piact_bBRS.png')
                torchvision.utils.save_image(iact_torch[0, 1], self.save_dir / f'{self.img_name}_niact_bBRS.png')
                torchvision.utils.save_image((segmap>0.5)*1.0, self.save_dir / f'{self.img_name}_pred_bBRS.png')

            # Check if BRS should be activated
            is_brs = torch.any(segmap[0,0][torch.tensor(self.pclick_map)] <= 0.5) or torch.any(segmap[0,0][torch.tensor(self.nclick_map)] > 0.5)
            if is_brs:
                # Get optimized data
                self.InputCorrection.init_click(image_nd, iact_torch, target_map, valid_map)
                self.opt_data = self.InputCorrection.optimize(1e-3, self.opt_data)

                # Perform forward pass with refined iact
                refined_iact = iact_torch + self.opt_data
                refined_segmap = self.net(image_nd, refined_iact)

                if self.save_iacts:
                    torchvision.utils.save_image(refined_iact[0, 0], self.save_dir / f'{self.img_name}_piact_aBRS.png')
                    torchvision.utils.save_image(refined_iact[0, 1], self.save_dir / f'{self.img_name}_niact_aBRS.png')
                    torchvision.utils.save_image((refined_segmap > 0.5) * 1.0, self.save_dir / f'{self.img_name}_pred_aBRS.png')

                return refined_segmap, refined_iact

        return segmap, iact_torch
