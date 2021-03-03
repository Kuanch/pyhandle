import unittest

import torch

import train


class InitArgments(object):
    def __init__(self):
        self.model_name = 'resnet18'
        self.dataset_name = 'CIFAR10'
        self.dataset_path = ''
        self.training_epoch = 1
        self.train_batch_size = 32
        self.test_batch_size = 128

    def load(self, json_path):
        raise NotImplementedError


class TestTrainSetUp(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        default_args = InitArgments()
        self.setup = train.training_setup(default_args)

    def _model_with_all_none_grad(self, model):
        for p in model.parameters():
            if p.data.grad is not None:
                return False

        return True

    def _loss_fn_with_none_weight(self, loss_fn):
        if loss_fn.weight is not None:
            return False

        return True

    def _optim_with_all_none_grad(self, optim):
        if 'params' in optim.param_groups[0]:
            for param in optim.param_groups[0]['params']:
                if param.requires_grad:
                    if param.grad is not None:
                        return False
            return True

        else:
            KeyError('No params field found in the optimizer.')

    def _dataset_train_loader_not_empty(self, dataset):
        if not hasattr(dataset, 'train_loader'):
            return False

        return True

    def test_init_model(self):
        model = self.setup['model']

        self.assertTrue(self._model_with_all_none_grad(model))

    def test_init_loss(self):
        loss_fn = self.setup['criterion']

        self.assertTrue(self._loss_fn_with_none_weight(loss_fn))

    def test_init_optim(self):
        optim = self.setup['optimizer']

        self.assertTrue(self._optim_with_all_none_grad(optim))

    def test_init_dataset(self):
        dataset = self.setup['dataset']

        self.assertTrue(self._dataset_train_loader_not_empty(dataset))


class TestTrainForOneStep(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        default_args = InitArgments()
        self.setup = train.training_setup(default_args)
        self.setup['loss_container'] = [0.]
        self.setup.pop('writer')

    def _if_param_change(self, init_params, params):
        for (_, p0), (name, p1) in zip(init_params, params):
            if torch.equal(p0, p1):
                return False

        return True

    def test_param_change(self):
        model = self.setup['model']
        trained_params = [np for np in model.named_parameters() if np[1].requires_grad]
        init_params = [(name, param.clone()) for (name, param) in trained_params]

        train.train_for_one_step(**self.setup)

        self.assertTrue(self._if_param_change(init_params, trained_params))


if __name__ == '__main__':
    unittest.main(verbosity=2)
