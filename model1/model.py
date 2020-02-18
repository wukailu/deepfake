import torchvision.models as models
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


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


def create_model(use_hidden_layer, dropout, backbone, params):
    if backbone == 1:
        model = models.resnet18(pretrained=True)
        model.fc = get_classifier(model.fc.in_features, use_hidden_layer, dropout)
        params['batch_size'] = 256
    elif backbone == 2:
        model = models.resnet34(pretrained=True)
        model.fc = get_classifier(model.fc.in_features, use_hidden_layer, dropout)
    elif backbone == 3:
        model = models.resnet50(pretrained=True)
        model.fc = get_classifier(model.fc.in_features, use_hidden_layer, dropout)
    elif backbone == 4:
        model = models.resnet101(pretrained=True)
        model.fc = get_classifier(model.fc.in_features, use_hidden_layer, dropout)
    elif backbone == 5:
        model = models.densenet121(pretrained=True)
        model.classifier = get_classifier(model.classifier.in_features, use_hidden_layer, dropout)
    elif backbone == 6:
        model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=23)
        model._dropout = nn.Dropout(0)
        model._fc = get_classifier(model._fc.in_features, use_hidden_layer, dropout)
    elif backbone == 7:
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=23)
        model._dropout = nn.Dropout(0)
        model._fc = get_classifier(model._fc.in_features, use_hidden_layer, dropout)
    elif backbone == 8:
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=23)
        model._dropout = nn.Dropout(0)
        model._fc = get_classifier(model._fc.in_features, use_hidden_layer, dropout)
        params['batch_size'] = 32
    else:
        print("Unrecognized model name, using resnet18")
        model = models.resnet18(pretrained=True)
        model.fc = get_classifier(model.fc.in_features, use_hidden_layer, dropout)

    # TODO: add this to hyper-parameter search
    freeze_until(model, "_blocks.4._expand_conv.weight")
    print(model)
    model = model.cuda()
    return model, params

