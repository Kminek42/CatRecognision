from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib.pylab as plt
import sys
import numpy as np


class NeuralNetwok:
    def __init__(self, shape: np.array, genotype: np.array):
        self.activation = np.tanh
        self.w = []
        self.genotype = genotype


        parameters_n = 0
        for i in range(len(shape) - 1):
            parameters_n += shape[i] * shape[i+1]
            parameters_n += shape[i+1]

        assert len(self.genotype) == parameters_n, f"genotype: {len(self.genotype)}, parameters: {parameters_n}"
            
        for i in range(len(shape) - 1):
            self.w.append(genotype[:shape[i] * shape[i+1]].reshape(shape[i + 1], shape[i]))
            genotype = genotype[shape[i] * shape[i+1]:]

            self.w.append(genotype[:shape[i+1]])
            self.w[-1] = self.w[-1].reshape(shape[i+1], 1)
            genotype = genotype[shape[i+1]:]

        
    def forward(self, inputs):
        inputs = inputs.reshape(1, -1).T
        for i in range(0, len(self.w), 2):
            inputs = self.w[i] @ inputs
            inputs += self.w[i + 1]
            inputs = self.activation(inputs)
        
        inputs = np.clip(inputs, -1, 1)
        return inputs.T[0]

# load model -------------------------------------------------------------------
activation = nn.Tanh()

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(768, 64),
    activation,
    nn.Linear(64, 32),
    activation,
    nn.Linear(32, 16),
    activation,
    nn.Linear(16, 2)
)

model.load_state_dict(torch.load('model.pt'))
model.eval()

parameters = []
for param in model.parameters():
    parameters.append(param.detach().numpy().flatten())

# Flatten the parameters into a single genotype array
genotype = np.concatenate(parameters)
layers = (768, 64, 32, 16, 2)
network = NeuralNetwok(layers, genotype)

for _ in range(10):
    inputs = np.random.randn(768)
    print(8 * '-')
    print(network.forward(inputs))
    print(model.forward(torch.Tensor(inputs).unsqueeze(0))[0].detach().numpy())


with open('genotype.csv', 'w') as f:
    np.savetxt(f, genotype)

with open('layers.csv', 'w') as f:
    for layer in layers[:-1]:
        f.write(str(layer) + ',')
    f.write(str(layers[-1]))