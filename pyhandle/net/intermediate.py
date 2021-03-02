import torch
import torch.nn as nn
from net.network import get_network


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IntermediateNetwork(nn.Module):
    def __init__(self, network_name, layers_idx, pretrained=True):
        super(IntermediateNetwork, self).__init__()
        net = get_network(network_name, pretrained).to(device)
        self.features = list(net.children())
        self.layers = layers_idx

    def forward(self, x):
        outputs = []
        for layer_idx, conv in enumerate(self.features):
            if layer_idx > max(self.layers):
                break
            x = conv(x).to(device)
            if layer_idx in self.layers:
                outputs.append(x)

        return outputs
