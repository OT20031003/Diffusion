from torch.nn import Module
import torch, math
from PIL import Image
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import random
from embedingver import ResNet, Downsample, Upsample, UNet, SinusoidalPositionEmbeddings, SmallUNet
import os
from torchvision.datasets import ImageFolder # ImageFolderをインポート
from DDPM import load_image_as_tensor, save_tensor_as_image, GaussianDiffusion




        
def Training(model, optimizer):
    model.train()
    dir_path = "./dataset" # dataset内のフォルダの画像
    epoch = 100
    sepoch = epoch
    files_file = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    while epoch >= 0:
        print(f"epoch = {sepoch - epoch}")
        for x in files_file:
            optimizer.zero_grad()
            print(x)
            x_0 = load_image_as_tensor(dir_path + "/" +x)
            x_0 = x_0.unsqueeze(0)
            x_0 = x_0 * 2 - 1
            maxx = model.timesteps
            minn = 0
            t = torch.empty(1).uniform_(minn, maxx).long()
            print(t)
            epsilon = torch.randn(x_0.shape)
            
            predict = model.unet.forward(model.forward_process(x_0, t, noise=epsilon), t)
            criterion = torch.nn.MSELoss()
            loss = criterion(epsilon , predict)
            print(f"loss = {loss}")
            loss.backward()
            optimizer.step()
        epoch -= 1
    torch.save(model.state_dict(), 'model_weight.pth')







def Training_test():
    model = GaussianDiffusion()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    Training(model, optimizer)


def main():
    print("code execute!!")
    Training_test()

main()