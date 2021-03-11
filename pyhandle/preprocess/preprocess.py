import torchvision.transforms as transforms


class CIFAR_PREPROCESS(object):
    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


dataset_preprocess_map = {'cifar10': CIFAR_PREPROCESS}


def get_preprocessor(name, kwargs={}):
    if name.lower() in dataset_preprocess_map:
        preprocessor = dataset_preprocess_map[name.lower()](**kwargs).transform

    return preprocessor
