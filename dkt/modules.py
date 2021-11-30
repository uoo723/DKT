import torch.nn as nn


class Identity(nn.Module):
    def forward(self, inputs):
        return inputs
