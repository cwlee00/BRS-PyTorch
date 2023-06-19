import torch
from scipy.optimize import fmin_l_bfgs_b

from utils import *
from BRS.LossFunctions import CorrectiveEnergy, InertialEnergy


class BRS:
    def __init__(self, net, device):
        self.net = net
        self.device = device
        self.CE = CorrectiveEnergy()
        self.IE = InertialEnergy()

    def init_click(self, img, iact, tr_tmat, tr_vmat):
        self.in_img = img
        self.b, self.h, self.w = img.shape[0], img.shape[2], img.shape[3]
        self.init_iact = iact

        self.tr_tmat = tr_tmat
        self.tr_vmat = tr_vmat

    def optimize(self, reg_param, opt_data):
        self.reg_param = reg_param
        result = fmin_l_bfgs_b(func=self._optimize_function, x0=opt_data.cpu().numpy().ravel(), m=20, factr=0, pgtol=1e-8, maxfun=20, maxiter=40)   #result[0]: optimized_iact, result[1]: f_val
        optimized_data = torch.from_numpy(result[0]).view(self.b, 2, self.h, self.w).float().to(self.device)

        return optimized_data

    def _optimize_function(self, x):
        opt_data = torch.from_numpy(x).float().to(self.device)
        opt_data.requires_grad_(True)

        with torch.enable_grad():
            # Forward
            refined_iact = self.init_iact + opt_data.view(self.b, 2, self.h, self.w)
            segmap = self.net(self.in_img, refined_iact)

            # Compute corrective energy
            loss_CE, f_max = self.CE(segmap, self.tr_tmat, self.tr_vmat)

            # Compute inertial energy
            loss_IE = self.IE(refined_iact, self.init_iact)

            # Compute total energy
            loss = torch.sum(loss_CE + (self.reg_param * loss_IE))

        f_val = loss.detach().cpu().numpy()

        if f_max < 0.5:
            f_grad = np.zeros_like(x)

        else:
            loss.backward()
            f_grad = opt_data.grad.cpu().numpy().ravel().astype(np.float)

        return [f_val, f_grad]