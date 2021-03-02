import torch
import torchvision.models as models


def get_network(network_name, pretrained=True):
    if hasattr(models, network_name):
        pretrained_model = getattr(models, network_name)(pretrained=pretrained)
        return pretrained_model

    else:
        raise ImportError('Unknown network {}'.format(network_name))


def get_classifier(network_name, num_classes=1000, pretrained=True):
    network_base = get_network(network_name, pretrained)

    if pretrained and num_classes != 1000:
        fully_connect = torch.nn.Linear(1000, num_classes)
        return torch.nn.Sequential(network_base, fully_connect)
