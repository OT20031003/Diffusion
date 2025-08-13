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
        self.beta_schedules = make_beta_schedule(timesteps, start_beta, end_beta)
        self.alpha_schedules = []
        for i in range(timesteps):
            self.alpha_schedules.append(1 - self.beta_schedules[i])
        self.alpha_bar_schedules = []
        tmp = 1
        for i in range(timesteps):
            self.alpha_bar_schedules.append(tmp * self.alpha_schedules[i])
            tmp *= self.alpha_schedules[i]
        #print(alpha_bar_schedules)
        self.schedule = torch.tensor(self.alpha_bar_schedules)
        self.register_buffer('betas', torch.tensor(self.beta_schedules))
        self.register_buffer('alphas', torch.tensor(self.alpha_schedules))
        self.register_buffer('alpha_bars', torch.tensor(self.alpha_bar_schedules))
        self.unet = UNet(3, 64, 128)
        self.timesteps = timesteps
        #print(type(self.schedule))
    
    def forward_process(self, img, timestep, noise = None):
        assert(type(img) == torch.Tensor)
        if noise == None:
            noise = torch.randn_like(img)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[timestep]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar  = torch.sqrt(1 - self.alpha_bars[timestep]).view(-1, 1, 1, 1)
        img2 = sqrt_alpha_bar * img + sqrt_one_minus_alpha_bar * noise
        return img2
    
    

    def reverse_onestep(self, img, timestep):
        """ 1ステップの逆拡散 """
        # timestep は torch.tensor([999]) のような形式で渡されることを想定
        
        # 予測されたノイズを取得
        epsilon_theta = self.unet.forward(img, timestep)
        
        # 各種パラメータを取得
        alpha_t = self.alphas[timestep]
        alpha_bar_t = self.alpha_bars[timestep]
        beta_t = self.betas[timestep]
        
        # ノイズを生成
        z = torch.randn_like(img)
        
        # 最後のステップ (t=0) ではノイズを加えない
        if timestep.item() == 0:
            z = torch.zeros_like(img)
            
        sigma_t = torch.sqrt(beta_t)
        
        # DDPMの論文に基づいた正しいサンプリング式に修正
        # 誤: 1/torch.sqrt(alpha_bar_t)
        # 正: 1/torch.sqrt(alpha_t)
        term1 = 1 / torch.sqrt(alpha_t)
        term2 = (img - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * epsilon_theta)
        
        return term1 * term2 + sigma_t * z

    def reverse_process(self, img, timestep):
        """ timestepから0になるまで逆拡散を繰り返す """
        
        # 正しい型チェック (isinstanceを使用)
        save_tensor_as_image(img.squeeze(0), "./result/ongo_start.png")
        if not isinstance(timestep, torch.Tensor):
            print(f"エラー: timestepはTensorである必要がありますが、{type(timestep)}が渡されました。")
            raise TypeError("timestep must be a torch.Tensor")

        # .item() を使ってTensorからPythonの数値を取得
        ts = timestep.item()

        # ts から 0 までループ
        # Python 3のrangeでは逆順のループは range(start, stop, step) を使う
        for current_t in range(ts-1, -1, -1):
            # 現在のタイムステップをTensorに変換して渡す
            current_t_tensor = torch.tensor([current_t], device=img.device)
            img = self.reverse_onestep(img, current_t_tensor)
            if current_t % 250 == 0 or current_t == ts - 1:
                print(f"current_t = {current_t}")
                save_tensor_as_image(img.squeeze(0), "./result/ongo" + str(current_t)+".png")
            
        return img



        
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

def InferTest():
    """
    学習済みモデルを使い、ノイズから画像を生成して保存する関数。
    """
    # 1. デバイスの設定（GPUが利用可能ならGPUを使用）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. モデルのインスタンス化と学習済み重みのロード
    model = GaussianDiffusion()
    
    # Training関数で保存されるファイル名 'model_weight.pth' を指定
    # map_location=device を使うことで、GPUがない環境でもGPUで学習したモデルを読み込める
    try:
        model.load_state_dict(torch.load('model_weight.pth', map_location=device))
    except FileNotFoundError:
        print("Error: 'model_weight.pth' not found.")
        print("Please train the model first by uncommenting and running the Training() function in main().")
        return
        
    model.to(device)
    model.eval() # モデルを評価モードに設定（Dropoutなどを無効化）

    # 3. 画像生成の準備
    # Training時の画像サイズ（CenterCrop(256)）に合わせる
    image_size = 256
    channels = 3
    batch_size = 1 # 一度に生成する画像の枚数

    # 4. 逆拡散プロセスによる画像生成
    print("Generating image from pure noise...")
    
    # (batch_size, channels, height, width) の形状でランダムノイズを生成
    img = torch.randn((batch_size, channels, image_size, image_size), device=device)
    save_tensor_as_image(img.squeeze(0), "rand.png")
    # 勾配計算は不要なため、torch.no_grad()コンテキストで実行
    with torch.no_grad():
        # model.timesteps - 1 から 0 までループ
        tss = torch.tensor([model.timesteps])
        img2 = model.reverse_process(img, tss)
        # for t in reversed(range(0, model.timesteps)):
        #     # 現在のタイムステップをモデル入力用のテンソル形式に変換
        #     timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
        #     # 1ステップ分のノイズ除去を実行し、画像を更新
        #     img = model.reverse_onestep(img, timestep)
    
    print("Image generation complete.")

    # 5. 生成した画像を保存
    # [-1, 1] の範囲で出力される画像を [0, 1] に変換し、PILで扱えるようにCPUに送る
    if torch.isnan(img2).any():
        print("NaN detected in generated image!")
    if torch.isinf(img2).any():
        print("Inf detected in generated image!")

    generated_image = (img2.clamp(-1, 1) + 1) / 2
    
    save_tensor_as_image(generated_image.squeeze(0).cpu(), "generated_image.png")

def Training_test():
    model = GaussianDiffusion()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    Training(model, optimizer)


def main():
    print("code execute!!")
    # model = GaussianDiffusion()
    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    Training_test()
    #InferTest()
main()