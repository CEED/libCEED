#!/usr/bin/env python3
import torch
import torch.nn as nn
# from IPython.core.debugger import set_trace
from pathlib import Path
import numpy as np


new_parameters_Path = Path('../dd_sgs_data')

weights = []
biases = []
weights.append(np.loadtxt(new_parameters_Path / 'w1.dat', skiprows=1).reshape(6,20).T)
weights.append(np.loadtxt(new_parameters_Path / 'w2.dat', skiprows=1).reshape(20,6).T)
biases.append(np.loadtxt(new_parameters_Path / 'b1.dat', skiprows=1))
biases.append(np.loadtxt(new_parameters_Path / 'b2.dat', skiprows=1))

### Anisotropic SGS model for LES developed by Aviral Prakash and John A. Evans at UCB
class anisoSGS(nn.Module): 
    # The class takes as inputs the input and output dimensions and the number of layers   
    def __init__(self, inputDim=6, outputDim=6, numNeurons=20, numLayers=1):
        super().__init__()
        self.ndIn = inputDim
        self.ndOut = outputDim
        self.nNeurons = numNeurons
        self.nLayers = numLayers
        self.net = nn.Sequential(
            nn.Linear(self.ndIn, self.nNeurons),
            nn.LeakyReLU(0.3),
            nn.Linear(self.nNeurons, self.ndOut))
        
    # Define the method to do a forward pass
    def forward(self, x):
        return self.net(x)

def load_n_trace_model(model_name):
    # Instantiate PT model and load state dict
    model = anisoSGS()
    model.load_state_dict(torch.load(f'{model_name}.pt', map_location=torch.device('cpu')))
    model.double()

    # Change individual model weights
    with torch.no_grad():
        # for layer in [0,2]:
        for i, layer in enumerate([0,2]):
            m,n = model.net[layer].weight.shape
            print('weight shape', m,n)
            # set_trace()

            model.net[layer].weight[...] = torch.from_numpy(weights[i])[...]
            model.net[layer].bias[...] = torch.from_numpy(biases[i])[...]

            # for i in range(m):
            #     for j in range(n):
            #         # print(model.net[layer].weight[i,j])
            #         model.net[layer].weight[i,j] = 1.0
            #         # print(model.net[layer].weight[i,j])

    # Prepare model for inference
    dummy_input = torch.randn(512, 6, dtype=torch.float64, device='cpu')
    with torch.no_grad():
        model_script = torch.jit.script(model)
        torch.jit.save(model_script, f"{model_name}_fp64_jit.ptc")

        model = torch.jit.trace(model, dummy_input)
        torch.jit.save(model, f"{model_name}_fp64_jit.pt")

    return model

def main():
    model_name = 'NNModel_HIT'
    model = load_n_trace_model(model_name)

if __name__ == '__main__':
    main()

