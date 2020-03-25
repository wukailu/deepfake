from model3.tsm.ops.models import TSN
from torch import nn


def print_model_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'total number of params: {pytorch_total_params:,}')
    return pytorch_total_params


def get_trainable_params(model):
    print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("\t", repr(name))
            params_to_update.append(param)
            
    return params_to_update


def create_model(params):
    backbone = params['backbone']
    if backbone == 0:
        model = TSN(num_class=1, num_segments=params["num_segments"], modality="RGB",
                    base_model="resnet18", dropout=params["dropout"], partial_bn=False,
                    is_shift=True, shift_div=params["shift_div"], fc_lr5=True)
    elif backbone == 1:
        model = TSN(num_class=1, num_segments=params["num_segments"], modality="RGB",
                    base_model="resnet50", dropout=params["dropout"], partial_bn=False,
                    is_shift=True, shift_div=params["shift_div"], fc_lr5=True)
    elif backbone == 2:
        model = TSN(num_class=1, num_segments=params["num_segments"], modality="RGB",
                    base_model="resnext50_32x4d", dropout=params["dropout"], partial_bn=False,
                    is_shift=True, shift_div=params["shift_div"], fc_lr5=True)
    else:
        raise NotImplementedError()

    # print(model)
    return model.cuda(), params


