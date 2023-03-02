# minist 用MLP实现，MLP也是使用pytorch实现的
import torchvision
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
import time

class Model(torch.nn.Module):

    def __init__(self, num_i, num_h, num_o):
        super(Model, self).__init__()

        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_h)  # 2个隐层
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h, num_o)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


