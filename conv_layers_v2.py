import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os, shutil
import matplotlib.pyplot as plt
torch_seed = 0
torch.manual_seed(torch_seed)

def create_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

ARTIFACTS_DIR = os.getcwd() + '/artifacts'
create_dir(ARTIFACTS_DIR)

def save_img(path, tensor):
    tensor = tensor.squeeze(0)
    if len(tensor.shape) > 2:
        tensor = tensor.mean(dim=0)
    tensor_np = tensor.detach().cpu().numpy()
    plt.axis('off')
    print(path)
    plt.imsave(path, tensor_np, cmap='gray')

class ConvLayersV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLayersV2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, bias=False)

    def forward(self, x):
        _conv1 = self.conv1(x)
        conv2_outputs = []
        for j in range(_conv1.shape[1]): # for each input channel
            conv2_output_i = []
            for i in range(self.conv2.weight.shape[0]): # for each output channel
                filter_weights = self.conv2.weight[i:i+1, j:j+1, :, :]
                conv2_output_ij = nn.functional.conv2d(_conv1[:, j:j+1, :, :], filter_weights)
                os.makedirs(ARTIFACTS_DIR + f'/out_channel_{i}', exist_ok=True)
                save_img(ARTIFACTS_DIR + f'/out_channel_{i}/out_{i}_in_{j}.jpeg', conv2_output_ij)
                conv2_output_i.append(conv2_output_ij)
            conv2_output_i_ = torch.cat(conv2_output_i, dim=1)  # torch.Size([1, 64, 568, 568]), Concatenate the outputs along the channel dimension
            conv2_outputs.append(conv2_output_i_) # contains N tensors of shape (batch, C, H, W) torch.Size([1, 64, 568, 568])
            print(100*'-')
        os.makedirs(ARTIFACTS_DIR + '/concat', exist_ok=True)
        for i, out_channel_output in enumerate(torch.stack(conv2_outputs, dim=0)):
            save_img(ARTIFACTS_DIR + f'/concat/out_channel_{i}.jpeg', out_channel_output)
        # Concatenate all output channels
        stack_conv2 = torch.stack(conv2_outputs) # torch.Size([64, 1, 64, 568, 568])
        _conv2 = torch.sum(stack_conv2, dim=0)  # Sum the all the concatenated intermediate results along the stacked dimension
        save_img(ARTIFACTS_DIR + '/conv2_output_v2.jpeg', _conv2)
        return _conv2 # torch.Size([1, 64, 568, 568])

if __name__ == "__main__":
    image = Image.open("cat_dog.jpeg")
    image_gray = image.convert("L")
    transform = transforms.Compose([
        transforms.Resize((572, 572)),          
        transforms.ToTensor(),                  
    ])

    sample_input = transform(image_gray).unsqueeze(0)
    model = ConvLayersV2(in_channels=1, out_channels=64)
    output = model(sample_input)
    print("Output shape:", output.shape) # torch.Size([1, 64, 568, 568])
    print(output)
