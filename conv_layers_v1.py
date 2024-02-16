import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
torch_seed = 0
torch.manual_seed(torch_seed)

def save_img(path, tensor):
    tensor = tensor.squeeze(0)
    if len(tensor.shape) > 2:
        tensor = tensor.mean(dim=0)
    tensor_np = tensor.detach().cpu().numpy()
    plt.axis('off')
    plt.imsave(path, tensor_np, cmap='gray')

class ConvLayersV1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLayersV1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, bias=False)

    def forward(self, x):
        # print(x.shape) # torch.Size([1, 1, 572, 572])
        _conv1 = self.conv1(x)
        # print(_conv1.shape) # torch.Size([1, 64, 570, 570])
        _conv2 = self.conv2(_conv1)
        # print(_conv2.shape) # torch.Size([1, 64, 568, 568])
        save_img(os.getcwd()+ '/conv2_output_v1.jpeg', _conv2)
        return _conv2

if __name__ == "__main__":
    # Open the image
    image = Image.open("cat_dog.jpeg")
    image_gray = image.convert("L")
    transform = transforms.Compose([
        transforms.Resize((572, 572)),         
        transforms.ToTensor(),                 
    ])

    sample_input = transform(image_gray).unsqueeze(0)
    model = ConvLayersV1(in_channels=1, out_channels=64)
    weights_conv1 = model.conv1.weight
    print('Weights Conv1: ', weights_conv1.shape, weights_conv1)
    weights_conv2 = model.conv2.weight
    print('Weights Conv2: ', weights_conv2.shape, weights_conv2)
    output = model(sample_input)
    print("Output shape:", output.shape) # torch.Size([1, 64, 568, 568])
    print(output)
