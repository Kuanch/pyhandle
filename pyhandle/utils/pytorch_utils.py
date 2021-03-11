import os
import torch


def save_model(model_state, optim_state, preprocessor, loss, epoch, path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    torch.save({'epoch': epoch,
                'model_state_dict': model_state,
                'optim_state_dict': optim_state,
                'preprocessor': preprocessor,
                'loss': loss}, path)
