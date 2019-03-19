import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# provide structure of your model so it can be imported together with its state
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        self.passf = nn.ReLU()

    def forward(self, x):
        return x
