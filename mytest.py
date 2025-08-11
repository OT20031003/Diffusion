from torch.nn import Module
import torch, math
from PIL import Image
from torchvision import transforms
import random
from embedingver import ResNet, Downsample, Upsample, UNet, SinusoidalPositionEmbeddings
def make_beta_schedule(timesteps, start_beta, end_beta):
    a = (end_beta - start_beta ) /timesteps
    #y = a * time + start_beta
    sch = []
    for t in range(timesteps):
        assert(a*t + start_beta < 1)
        sch.append(a * t + start_beta)
    return sch  

class GaussianDiffusion(Module):
    def __init__(
            self, 
            timesteps = 1000, 
            start_beta = 0.01,
            end_beta = 0.22
    ):
        super().__init__()
        beta_schedules = make_beta_schedule(timesteps, start_beta, end_beta)
        alpha_schedules = []
        for i in range(timesteps):
            alpha_schedules.append(1 - beta_schedules[i])
        alpha_bar_schedules = []
        tmp = 1
        for i in range(timesteps):
            alpha_bar_schedules.append(tmp * alpha_schedules[i])
            tmp *= alpha_schedules[i]
        #print(alpha_bar_schedules)
        self.schedule = torch.tensor(alpha_bar_schedules)
        #print(type(self.schedule))
    
    def forward_process(self, img, timestep):
        assert(type(img) == torch.Tensor)
        noise = torch.randn_like(img)
        img2 = torch.sqrt(self.schedule[timestep]) * img + torch.sqrt(1 - self.schedule[timestep]) *noise
        return img2
    
    def reverse_process(self, img, timesteps):
        pass
     
def load_image_as_tensor(image_path:str)->torch.Tensor:
    try:
        pil_img = Image.open(image_path)
        # 256にクリップ
        transform_clip = transforms.CenterCrop(256)
        transform = transforms.ToTensor()
        tensor_img = transform(pil_img)
        tensor_img = transform_clip(tensor_img)

        return tensor_img
    except FileNotFoundError:
        print(f"The file at {image_path} was not found")

def save_tensor_as_image(tensor: torch.Tensor, save_path: str):
    try:
        # DDPMの出力が[-1, 1]の場合、[0, 1]に正規化する
        # (x + 1) / 2 は [-1, 1]の値を [0, 1]に変換する一般的な方法です。
        if tensor.min() >= -1.0 and tensor.max() <= 1.0:
            tensor = (tensor + 1) / 2.0
            
        # ToPILImage()は、(C, H, W)のテンソルをPIL画像オブジェクトに変換します。
        # 入力テンソルの値は、0.0-1.0の範囲である必要があります。
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(tensor)
        
        # PILのsaveメソッドで画像を保存
        pil_image.save(save_path)
        print(f"Image successfully saved to {save_path}")
        
    except Exception as e:
        print(f"An error occurred while saving the image: {e}")

def main():
    g = GaussianDiffusion()
    resnet = ResNet(3, 128)
    down = Downsample(3)
    up = Upsample(3)
    time_emb_layer = SinusoidalPositionEmbeddings(dim=128)

    #u = UNet(3, 64)
    img_path = "./dataset/02.jpg"
    inp_img = load_image_as_tensor(img_path)
    inp_img = 2*(inp_img) - 1
    save_tensor_as_image(inp_img, "./a.png")
    inp_img = inp_img.unsqueeze(0) # バッチ追加
    # ここから[1,3,256,256]
    u = UNet(3, 64, timeembedding_dim=128)
    uf = u.forward(inp_img, torch.Tensor([10]))
    save_tensor_as_image(uf.squeeze(0), "./c.png")
main()