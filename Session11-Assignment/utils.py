import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import time
import sys
import matplotlib.pyplot as plt
import os
from gradcam import GradCAM, GradCAMpp
from gradcam.utils import visualize_cam
from torchvision.utils import make_grid, save_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    pass
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


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

# _, term_width = os.popen('stty size', 'r').read().split()
def get_terminal_size():
    import shutil
    return shutil.get_terminal_size((80, 20))

term_width, _ = get_terminal_size()

term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def display_results(results):
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    fig = plt.figure(figsize=(10,10))

    # Only take first 9 results
    for i in range(9):
        img, actual, predicted = results[i]

        ax = fig.add_subplot(3, 3, i+1)
        ax.imshow(torchvision.utils.make_grid(torch.Tensor(img)).permute(1, 2, 0))
        ax.axis('off')
        ax.set_title(f"Actual: {classes[actual]} | Predicted: {classes[predicted]}")
    
    plt.tight_layout()
    plt.show()


def plot_losses(train_losses, test_losses):

    plt.figure(figsize=(12, 6))

    # Plot train loss
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')

    # Plot test loss
    plt.plot(range(len(test_losses)), test_losses, label='Test Loss')

    plt.title('Train and Test Loss over time')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


from torchvision.utils import make_grid

def generate_gradcam_images(model, data_loader, device, num_images=10):
    incorrect_samples = []
    model.eval()
    
    # Iterate over the test data
    for data, target in data_loader:
        # Move the data to the correct device
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = model(data)
        
        # Get the predicted classes
        pred = output.argmax(dim=1, keepdim=True)
        
        # Find the incorrectly classified samples
        incorrect_idx = ~pred.eq(target.view_as(pred)).squeeze()
        incorrect_data = data[incorrect_idx]
        incorrect_target = target[incorrect_idx]
        incorrect_pred = pred[incorrect_idx]
        
        # Save the incorrectly classified samples
        for data, target, pred in zip(incorrect_data, incorrect_target, incorrect_pred):
            # Check if num_images samples have already been collected
            if len(incorrect_samples) >= num_images:
                break

            incorrect_samples.append((data.cpu().numpy(), target.cpu().numpy(), pred.cpu().numpy()))
        
        # Exit the loop once num_images samples have been collected
        if len(incorrect_samples) >= num_images:
            break
    
    gradcam = GradCAM(model, target_layer=model.module.layer4[-1])
    
    fig, axarr = plt.subplots(nrows=num_images, ncols=2)
    
    for idx, (img, actual, predicted) in enumerate(incorrect_samples):
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)  # Convert numpy array to tensor
        heatmap, result = gradcam(img_tensor, class_idx=predicted)
        
        img = np.transpose(img, (1, 2, 0))  # Change (C, H, W) to (H, W, C) for displaying the image
        
        axarr[idx, 0].imshow(img, cmap="gray")
        axarr[idx, 0].set_title(f"Predicted: {predicted}, Actual: {actual}")
        
        axarr[idx, 1].imshow(transforms.ToPILImage()(result), cmap="gray")
        axarr[idx, 1].set_title("GradCAM")
    
    plt.show()


# generate_gradcam_images(net, testloader, device)
