import numpy as np
import torch
import torchvision
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy

from torch.utils import data
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import time

cuda = torch.cuda.is_available()
print("CUDA is ", cuda)
device = torch.device("cuda" if cuda else "cpu")

number_linear_layers = 2
