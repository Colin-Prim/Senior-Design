import numpy as np

def load_data(path):
    return np.load(path)

def save_model(model, path):
    torch.save(model.state_dict(), path)
