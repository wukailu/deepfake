import torch
import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


def visualize_metrics(records, extra_metric, name):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 6))
    axes[0].plot(list(range(len(records.train_losses))), records.train_losses, label='train')
    axes[0].plot(list(range(len(records.train_losses_wo_dropout))), records.train_losses_wo_dropout,
                 label='train w/o dropout')
    axes[0].plot(list(range(len(records.val_losses))), records.val_losses, label='val')
    axes[0].set_title('loss')
    axes[0].legend()

    axes[1].plot(list(range(len(records.train_accs))), records.train_accs, label='train')
    axes[1].plot(list(range(len(records.train_accs_wo_dropout))), records.train_accs_wo_dropout,
                 label='train w/o dropout')
    axes[1].plot(list(range(len(records.val_accs))), records.val_accs, label='val')
    axes[1].axhline(y=0.5, color='g', ls='--')
    axes[1].axhline(y=0.667, color='r', ls='--')
    axes[1].set_title('acc')
    axes[1].legend()

    axes[2].plot(list(range(len(records.train_custom_metrics))), records.train_custom_metrics, label='train')
    axes[2].plot(list(range(len(records.train_custom_metrics_wo_dropout))), records.train_custom_metrics_wo_dropout,
                 label='train w/o dropout')
    axes[2].plot(list(range(len(records.val_custom_metrics))), records.val_custom_metrics, label='val')
    axes[2].axhline(y=0.5, color='g', ls='--')
    axes[2].axhline(y=0.5, color='r', ls='--')
    axes[2].set_title(f'{extra_metric.__name__}')
    axes[2].legend()

    axes[3].plot(list(range(len(records.lrs))), records.lrs)
    _ = axes[3].set_title('lr')
    plt.tight_layout()
    plt.savefig(name, format='png')
    plt.close(fig)


def display_predictions_on_image(model, data, name):
    # val
    model.eval()

    inputs, labels = get_input_with_label(data)
    img_files = data['real_file'] + data['fake_file']

    with torch.no_grad():
        outputs = model(inputs)
        outputs_predicbilty = outputs  # torch.nn.functional.softmax(outputs, dim=1)
        assert len(outputs_predicbilty) == len(outputs), f'proba shape: {len(outputs_predicbilty)}'

        _, predicted = torch.max(outputs.data, 1)

    numbers = min(labels.size(0), 100)
    nrows = int(numbers ** 0.5)
    ncols = int(np.ceil(numbers / nrows))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 40))
    step = 0
    for i in range(nrows):
        for j in range(ncols):
            face_crop = np.load(img_files[step])
            axes[i, j].set_title(
                f'{outputs_predicbilty[step][0]:.2f},{outputs_predicbilty[step][1]:.2f}|{predicted[step]}|{labels[step]}')
            axes[i, j].imshow(face_crop)
            step += 1
            if step == numbers:
                break
    plt.title('predicted probability real, fake | prediction | label (0: real 1: fake)')
    plt.tight_layout()
    plt.savefig(name, format='png')
    plt.close(fig)


def get_input_with_label(data: dict, smooth=0):
    batch_size = data['real'].shape[0]
    inputs = torch.cat((data['real'], data['fake'])).cuda()
    labels = torch.cat((torch.zeros(batch_size), torch.ones(batch_size)))
    if smooth != 0:
        mask = (torch.randint_like(labels, 0, int(1 / smooth) + 1) // int(1 / smooth)).bool()
        labels[mask] = 1 - labels[mask]
    labels = labels.unsqueeze(dim=1).cuda()
    return inputs, labels


def multigpu_tensor_sum(tensor):
    import torch.distributed as dist
    rt = tensor.clone()  # The function operates in-place.
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def _gradient_penalty(net, real_data, generated_data):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda()
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = net(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(
                               prob_interpolated.size()).cuda(),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1).mean().data[0]

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return ((gradients_norm - 1) ** 2).mean()  # loss = loss + gp * lambda
