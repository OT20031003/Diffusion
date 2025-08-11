import torch 
import torch.nn as nn
from einops.layers.torch import Rearrange
import math

class SinusoidalPositionEmbeddings(torch.nn.Module):
    def __init__(self, dim):
        """
        Args:
            dim (int): 埋め込みベクトルの次元数
        """
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        https://www.nomuyu.com/positional-encoding/
        Args:
            time (torch.Tensor): タイムステップのバッチ。shapeは (Batch, )
        
        Returns:
            torch.Tensor: 時間埋め込みベクトル。shapeは (Batch, dim)
        """
        # 現在の計算デバイス（CPU or GPU）を取得
        device = time.device
        
        # 埋め込み次元の半分
        half_dim = self.dim // 2
        
        # 埋め込みの周波数を計算 (logスケールで)
        # 数式: 1 / (10000^(2i / dim))
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # タイムステップと周波数を掛け合わせる
        # time: (B,) -> (B, 1)
        # embeddings: (D/2,) -> (1, D/2)
        # 結果: (B, D/2) (ブロードキャストを利用)
        embeddings = time.unsqueeze(1) * embeddings.unsqueeze(0)
        
        # sinとcosを計算し、連結する
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        return embeddings


class ResNet(torch.nn.Module):
    def __init__(self, channels, timeembedding_dim):
        """ 
        timeembedding_dim: 埋め込みベクトルの次元
        """
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm = torch.nn.BatchNorm2d(channels)

        self.mlp = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(timeembedding_dim, channels)
        )
    def forward(self, x, timeembedding):
        h = x
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += h
        x = self.batch_norm(x)
        x = self.relu(x)
        
        x += self.mlp(timeembedding).unsqueeze(-1).unsqueeze(-1)
        return x
def default(x, y):
    if x == None:
        return y
    return x
class Upsample(nn.Module):
    def __init__(self, dim, out_dim=None):
        super().__init__()
        self.fn = nn.Upsample(scale_factor=2, mode = 'nearest')
        self.fn2 = nn.Conv2d(in_channels=dim, out_channels=default(out_dim, dim), kernel_size=3, padding=1)

    def forward(self, x):
        x = self.fn(x)
        x = self.fn2(x)
        return x

class Downsample(nn.Module):
    def __init__(self, dim, out_dim=None):
        super().__init__()
        # rearrでチャネル数を4倍にした後, Convで元のチャネル数に戻す
        self.rearr = Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2)
        self.conv = nn.Conv2d(in_channels=4*dim,out_channels=default(out_dim, dim), kernel_size=3,padding=1)
    
    def forward(self, x):
        x = self.rearr(x)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, dim, model_channel=64):
        super().__init__()
        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv_0 = nn.Conv2d(dim, model_channel, 3, 1, padding=1)
        dim = model_channel
        self.resnet1_1 = ResNet(dim)
        self.resnet1_2 = ResNet(dim)
        self.downsample1 = Downsample(dim, dim*2)
        dim *= 2
        self.resnet2_1 = ResNet(dim)
        self.resnet2_2 = ResNet(dim)
        self.downsample2 = Downsample(dim, dim * 2)
        dim *= 2
        self.resnet3_1 = ResNet(dim)
        self.resnet3_2 = ResNet(dim)
        self.downsample3 = Downsample(dim, dim * 2)
        dim *=2
        self.resnet4_1 = ResNet(dim)
        self.resnet4_2 = ResNet(dim)
        self.downsample4 = Downsample(dim, dim * 2)
        dim *=2
        #print(f"down = {dim}") # 1024
        self.resnet5_1 = ResNet(dim)
        self.resnet5_2 = ResNet(dim)
        self.upsample1 = Upsample(dim, dim//2)
        dim //= 2
        # dim = 512
        self.resnet6_1 = ResNet(dim*2)
        self.resnet6_2 = ResNet(dim*2)
        self.upsample2 = Upsample(dim*2, dim//2)
        dim //= 2
        # dim = 256
        self.resnet7_1 = ResNet(dim*2)
        self.resnet7_2 = ResNet(dim*2)
        self.upsample3 = Upsample(dim*2, dim//2)
        dim //= 2
        #dim = 128
        self.resnet8_1 = ResNet(dim*2)
        self.resnet8_2 = ResNet(dim*2)
        self.upsample4 = Upsample(dim*2, dim // 2)
        dim //= 2
        self.resnet9_1 = ResNet(dim*2)
        self.resnet9_2 = ResNet(dim*2)
        #print(f" dim = {dim}")
        assert(model_channel == dim)
        self.conv_last = nn.Conv2d(dim*2, 3, 3, 1, padding=1)

        


    def forward(self, x):
        x = self.conv_0(x)
        x = self.resnet1_1(x)
        #x = self.resnet1_2(x)
        x1 = x
        x = self.downsample1(x)
        x = self.resnet2_1(x)
        #x = self.resnet2_2(x)
        x2 = x
        x = self.downsample2(x)
        x = self.resnet3_1(x)
        #x = self.resnet3_2(x)
        x3 = x
        x = self.downsample3(x)
        x = self.resnet4_1(x)
        #x = self.resnet4_2(x)
        x4 = x
        x = self.downsample4(x)
        x = self.resnet5_1(x)
        x = self.upsample1(x)
        x = torch.cat((x4, x), dim = 1)
        x = self.resnet6_1(x)
        x = self.upsample2(x)
        x = torch.cat((x3, x), dim = 1)
        x = self.resnet7_1(x)
        x = self.upsample3(x)
        x = torch.cat((x2, x), dim = 1)
        x = self.resnet8_1(x)
        x = self.upsample4(x)
        x = torch.cat((x1, x), dim = 1)
        x = self.resnet9_1(x)
        x = self.conv_last(x)
        return x


        






