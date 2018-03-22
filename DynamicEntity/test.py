import torch
import torch.nn as nn
from torch.autograd import Variable

m = nn.Bilinear(20, 30, 1)
input1 = Variable(torch.randn(1, 20))
input2 = Variable(torch.randn(1, 30))
output = m(input1, input2)
print(output.size())