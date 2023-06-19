import torch


class CorrectiveEnergy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, activations, target_matrix, valid_matrix):
        valid_activations = torch.multiply(activations, valid_matrix)
        valid_target = torch.multiply(target_matrix, valid_matrix)
        loss = torch.sum(torch.pow((valid_activations - valid_target), 2.0))

        with torch.no_grad():
            f_max = torch.max(torch.abs(valid_activations - valid_target)).item()

        return loss, f_max


class InertialEnergy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, iact_map, init_map):
        loss = torch.sum((iact_map - init_map) ** 2)

        return loss
