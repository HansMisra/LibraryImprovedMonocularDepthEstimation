#load_model module

import torch
from model import DepthNet, create_dnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(path, model_type='depthnet', device=None):
    """
    Load the model from a specified path.
    
    Args:
        path (str): Path to the model's state dict.
        model_type (str, optional): Type of model to load. Defaults to 'depthnet'.
        device (torch.device, optional): Device to load the model onto. Defaults to None.
    
    Returns:
        torch.nn.Module: Loaded model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == 'depthnet':
        model = DepthNet().to(device)
    else:
        model = create_dnn().to(device)

    model.load_state_dict(torch.load(path))
    model.eval()
    return model
