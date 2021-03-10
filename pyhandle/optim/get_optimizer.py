import torch


APEX_SUPPORTED = ['SGD', 'Adam']


def get_optim(name, params, lr, momentum=0.9, betas=(0.9, 0.999)):
    enable_apex = False
    if name in APEX_SUPPORTED:
        try:
            import apex
            enable_apex = True
            print('Apex fused optim is enabled')
        except ImportError as e:
            print('Not using apex as it is not installed, error: ', e)

    if enable_apex:
        optim = getattr(apex.optimizers, 'Fused' + name)(params, lr)
    else:
        optim = getattr(torch.optim, name)(params, lr)

    return optim
