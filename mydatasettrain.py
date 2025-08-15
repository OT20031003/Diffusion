from torch.nn import Module
import torch, math
from PIL import Image
from torchvision import transforms, datasets
import torchvision
from torch.utils.data import DataLoader
import random
from embedingver import ResNet, Downsample, Upsample, UNet, SinusoidalPositionEmbeddings, SmallUNet
import os
from torchvision.datasets import ImageFolder # ImageFolderをインポート
from DDPM import load_image_as_tensor, save_tensor_as_image, GaussianDiffusion
import SmallDDPM


def MNISTtraining(model, optimizer):
    model.train()
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # <-- この行を追加
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5, ))
    ])
    train_dataset = datasets.MNIST(root='./MNISTdata', train=True, download=True, transform=transform)
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"train dataset size = {len(train_dataset)}")
    num_epoch = 3
    criterion = torch.nn.MSELoss()

    for epoch in range(num_epoch):
        for i, (images, _) in enumerate(train_loader):
            # images の shape: torch.Size([batch_size, 1, 28, 28])
            optimizer.zero_grad()
            maxx = model.timesteps
            minn = 0
            t =  torch.randint(minn, maxx, (batch_size,), dtype=torch.long)

            
            epsilon = torch.randn_like(images)
            #print(f"images.shape = {images.shape}, t.shape = {t.shape}")
            predict = model.unet.forward(model.forward_process(images, t, noise=epsilon), t)
            loss = criterion(epsilon, predict)
            if (i + 1) % 50 == 0:
                
                print(f"Epoch [{epoch + 1}/{num_epoch}], Step [{i+1}/ {len(train_loader)}]")
                print(f"loss = {loss}")
                if i > 300:
                    break
                
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), 'smalmodel_weight.pth')

                
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
    model = SmallDDPM.GaussianDiffusion()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
    MNISTtraining(model, optimizer)


def main():
    print("code execute!!")
    Training_test()

main()