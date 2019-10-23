import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

# num_parameters (int) – number of aa to learn. Although it takes an int as input, 
# there is only two values are legitimate: 1, or the number of channels at input. Default: 1
# init (float) – the initial value of aa. Default: 0.25
prelu = nn.PReLU(num_parameters=3, init=0.5)
input = autograd.Variable(torch.randn(1, 3, 224, 224)) * 100
print(input.shape)
print(input)

output = prelu(input)

print(output.shape)
print(output)

torch.onnx.export(prelu, input, "prelu.onnx")
