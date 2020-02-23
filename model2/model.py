import segmentation_models_pytorch as smp
import torchvision.models as models
import torch.nn as nn
import torch


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
    if backbone == 1:
        model: nn.Module = smp.Unet(encoder_name="resnet34", classes=1)
        # model = models.segmentation.fcn_resnet50(pretrained=True, num_classes=1)
    # elif backbone == 2:
    #     model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=True)
    else:
        raise Exception("Unrecognized model name, using resnet18")

    print(model)
    model = model.cuda()
    return model, params

