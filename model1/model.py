import torchvision.models as models
import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet
from numpy.random import choice


def check_model_block(model):
    for name, child in model.named_children():
        print(name)


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


def freeze_until(net, param_name):
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name


class MyResNeXt(models.resnet.ResNet):
    def __init__(self, pre_trained=True):
        super(MyResNeXt, self).__init__(block=models.resnet.Bottleneck,
                                        layers=[3, 4, 6, 3],
                                        groups=32,
                                        width_per_group=4)
        if pre_trained:
            checkpoint = torch.load("../input/pretrained-pytorch/resnext50_32x4d-7cdf4587.pth")
            self.load_state_dict(checkpoint)


def get_classifier(in_features, use_hidden_layer, dropout):
    if use_hidden_layer:
        return  nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.BatchNorm1d(in_features // 2),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, 1)
        )

    else:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )


def freeze_resnet(model, params):
    freeze = params['freeze']
    if freeze != 0:
        freeze = choice([2, 4])
        freeze_until(model, f"layer{freeze}.0.conv1.weight")
    params['freeze'] = int(freeze)
    return model, params


def create_model(use_hidden_layer, dropout, backbone, params):
    if backbone == 1:
        model = models.resnet18(pretrained=True)
        model.fc = get_classifier(model.fc.in_features, use_hidden_layer, dropout)
        params['batch_size'] = 256
        model, params = freeze_resnet(model, params)
    elif backbone == 2:
        model = models.resnet34(pretrained=True)
        model.fc = get_classifier(model.fc.in_features, use_hidden_layer, dropout)
        model, params = freeze_resnet(model, params)
    elif backbone == 3:
        model = models.resnet50(pretrained=True)
        model.fc = get_classifier(model.fc.in_features, use_hidden_layer, dropout)
        model, params = freeze_resnet(model, params)
    elif backbone == 4:
        model = models.resnet101(pretrained=True)
        model.fc = get_classifier(model.fc.in_features, use_hidden_layer, dropout)
        model, params = freeze_resnet(model, params)
        # 1 or 2 is better than 0
    elif backbone == 5:
        model = models.densenet121(pretrained=True)
        model.classifier = get_classifier(model.classifier.in_features, use_hidden_layer, dropout)
        # TODO: update freeze
        params['freeze'] = 0
    elif backbone == 6:
        model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=23)
        model._dropout = nn.Dropout(0)
        model._fc = get_classifier(model._fc.in_features, use_hidden_layer, dropout)
        # TODO: update freeze
        params['freeze'] = 0
    elif backbone == 7:
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=23)
        model._dropout = nn.Dropout(0)
        model._fc = get_classifier(model._fc.in_features, use_hidden_layer, dropout)
        if params['freeze'] != 0:
            freeze = choice(list(range(5, 15)))
            freeze_until(model, f"_blocks.{freeze}._expand_conv.weight")
            params['freeze'] = int(freeze)
    elif backbone == 8:
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=23)
        model._dropout = nn.Dropout(0)
        model._fc = get_classifier(model._fc.in_features, use_hidden_layer, dropout)
        params['batch_size'] = params['batch_size'] // 2
        # TODO: update freeze
        params['freeze'] = 0
    elif backbone == 9:
        model = models.resnext50_32x4d(pretrained=True)
        model.fc = get_classifier(model.fc.in_features, use_hidden_layer, dropout)
        model, params = freeze_resnet(model, params)
    else:
        print("Unrecognized model name, using resnet18")
        model = models.resnet18(pretrained=True)
        model.fc = get_classifier(model.fc.in_features, use_hidden_layer, dropout)
        model, params = freeze_resnet(model, params)

    print(model)
    model = model.cuda()
    return model, params

