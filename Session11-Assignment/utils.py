import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

# Image Transforms
def image_transforms():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return transform_train, transform_test


# Grad-CAM
def gradcam(model, input, class_idx, feature_module, target_layer_names):
    # Implementation depends on the model and layers you want to visualize
    # Please refer to https://github.com/jacobgil/pytorch-grad-cam for full Grad-CAM implementation


# Misclassifications
def get_misclassified_images(model, testloader):
    misclassified_images = []
    misclassified_labels = []
    misclassified_predictions = []

    model.eval()
    with torch.no_grad():
        for data, targets in testloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            mask = (predicted != targets)
            misclassified_images.append(data[mask])
            misclassified_labels.append(targets[mask])
            misclassified_predictions.append(predicted[mask])
    
    return misclassified_images, misclassified_labels, misclassified_predictions


# TensorBoard utility
def tb_writer(log_dir):
    writer = SummaryWriter(log_dir)
    return writer


# Advanced Training Policies
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=45):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
