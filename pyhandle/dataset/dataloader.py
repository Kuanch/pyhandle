import torch
import torchvision.datasets
import torchvision.transforms as transforms


class TorchLoader(object):
    def __init__(self, dataset_name,
                 transform=None,
                 train_batch_size=16,
                 test_batch_size=8,
                 data_path=None,
                 num_gpu=1):
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])

        if not isinstance(transform, transforms.Compose):
            raise TypeError('Preprocessor must be transforms.Compose type!')

        if hasattr(torchvision.datasets, dataset_name):

            train_set = getattr(torchvision.datasets, dataset_name)(root='./data',
                                                                    train=True,
                                                                    download=True,
                                                                    transform=transform)
            self.train_loader = torch.utils.data.DataLoader(train_set,
                                                            batch_size=train_batch_size,
                                                            shuffle=True,
                                                            num_workers=4 * num_gpu,
                                                            pin_memory=True)
            test_set = getattr(torchvision.datasets, dataset_name)(root='./data',
                                                                   train=False,
                                                                   download=True,
                                                                   transform=transform)
            self.test_loader = torch.utils.data.DataLoader(test_set,
                                                           batch_size=test_batch_size,
                                                           shuffle=True,
                                                           num_workers=4 * num_gpu,
                                                           pin_memory=False)

        else:
            raise ImportError('unknown dataset {}, only torchvision seems not to support'.format(dataset_name))
