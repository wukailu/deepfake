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


def get_classifier(in_features, use_hidden_layer, dropout):
    if use_hidden_layer:
        return  nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.BatchNorm1d(in_features // 2),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, 2),
            nn.Softmax(dim=1)
        )

    else:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 2),
            nn.Softmax(dim=1)
        )


def create_model(params):
    backbone = params['backbone']
    if backbone == 1:
        model = models.detection.maskrcnn_resnet50_fpn(pretrained=True, num_classes=1)
    # elif backbone == 2:
    #     model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=True)
    else:
        raise Exception("Unrecognized model name, using resnet18")

    print(model)
    model = model.cuda()
    return model, params

