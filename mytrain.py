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
    
    def forward_process(self, img, timestep):
        assert(type(img) == torch.Tensor)
        noise = torch.randn_like(img)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[timestep]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar  = torch.sqrt(1 - self.alpha_bars[timestep]).view(-1, 1, 1, 1)
        img2 = sqrt_alpha_bar * img + sqrt_one_minus_alpha_bar * noise
        return img2
    
    def bis_alpha(self, timestep):
        # timestep : tensor
        # index を返す
        ts = float(timestep[0])
        left = 0
        right = len(self.alpha_schedules) 
        while right - left > 1:
            mid = right + left
            mid //= 2
            if ts < self.alpha_schedules[mid]:
                left = mid
            elif ts > self.alpha_schedules[mid]:
                right = mid
            else:
                return mid
        return left
    
    def bis_beta(self, timestep):
        # timestep : tensor
        # index を返す
        ts = float(timestep[0])
        left = -1
        right = len(self.beta_schedules) - 1
        while right - left > 1:
            mid = right + left
            mid //= 2
            if ts > self.beta_schedules[mid]:
                left = mid
            elif ts < self.beta_schedules[mid]:
                right = mid
            else:
                return mid
        return right

    def bis_alph_bar(self, timestep):
        # timestep : tensor
        # index を返す
        ts = float(timestep[0])
        left = 0
        right = len(self.alpha_bar_schedules)
        while right - left > 1:
            mid = right + left
            mid //= 2
            if ts < self.alpha_bar_schedules[mid]:
                left = mid
            elif ts > self.alpha_bar_schedules[mid]:
                right = mid
            else:
                return mid
        return left

    def reverse_onestep(self, img, timestep):
        # 1ステップの逆拡散
        t = int(timestep[0])
        epsilon_theta = self.unet.forward(img, timestep)
        alpha_t = self.alphas[t] #バッファ
        alpha_bar_t = self.alpha_bars[t]
        z = torch.randn_like(img) # deviceも同じ
        if t == 0:
            z = torch.zeros_like(img) # 最後のステップはノイズｗ加えない
        beta_t = self.betas[t]
        sigma_t = torch.sqrt(beta_t)
        return (1/(torch.sqrt(alpha_bar_t))) * (img - ((1 - alpha_t)/ (torch.sqrt(1 - alpha_bar_t))) * epsilon_theta ) + sigma_t *z
    


    def reverse_process(self, img, timestep):
        # timestepはtensor([10])の形式
        # timestep->0になるまで逆拡散する
        ts = timestep[0]
        while ts >= 0:
            img = self.reverse_onestep(img, ts)
            ts -= 1

def get_ffhq_dataloader(image_path, image_size, batch_size):
    """
    FFHQデータセット用のデータローダーを作成する関数
    """
    transform = transforms.Compose([
        transforms.Resize(image_size), # 画像をリサイズ
        transforms.CenterCrop(image_size), # 中央をクロップ
        transforms.ToTensor(), # テンソルに変換
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [-1, 1]に正規化
    ])

    # ImageFolderを使ってデータセットを読み込む
    # image_pathはダウンロードした画像が格納されているディレクトリを指定
    dataset = ImageFolder(root=image_path, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader
def TrainingFFHQ(model, optimizer, num_epochs=60): # 関数名を変更
    model.train()
    device = next(model.parameters()).device
    
    # FFHQのデータローダーを取得
    # image_pathには、実際に画像を保存したディレクトリのパスを指定してください
    ffhq_loader = get_ffhq_dataloader(
        image_path="/home/naaa/.cache/kagglehub/datasets/tommykamaz/faces-dataset-small/versions/1/",  # 例: ./ffhq/images ディレクトリ
        image_size=256,             # 今回は256x256で試す
        batch_size=16
    )

    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch + 1}/{num_epochs} ---")
        # ImageFolderは(画像, ラベル)を返すので、ラベルは使わない
        for i, (images, _) in enumerate(ffhq_loader):
            
            print(f"i = {i}")
            optimizer.zero_grad()
            
            # 画像をモデルと同じデバイスに送る
            x_0 = images.to(device)

            # タイムステップとノイズも同じデバイス上に作成
            t = torch.randint(0, model.timesteps, (x_0.shape[0],), device=device).long()
            epsilon = torch.randn_like(x_0)

            # ノイズ画像を生成
            noisy_image = model.forward_process(x_0, t)
            # ノイズを予測
            predicted_noise = model.unet.forward(noisy_image, t)
            
            criterion = torch.nn.MSELoss()
            loss = criterion(epsilon, predicted_noise)
            
            loss.backward()
            optimizer.step()

            if (i + 1) %10  == 0:
                print(f"Batch [{i+1}/{len(ffhq_loader)}], Loss: {loss.item():.6f}")
                break
    torch.save(model.state_dict(), 'model_weight_ffhq.pth')
    print("Training finished and model saved.")
        
# def Training(model, optimizer):
#     model.train()
#     dir_path = "./dataset" # dataset内のフォルダの画像

#     files_file = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
#     for x in files_file:
#         optimizer.zero_grad()
#         print(x)
#         x_0 = load_image_as_tensor(dir_path + "/" +x)
#         x_0 = x_0.unsqueeze(0)
#         x_0 = x_0 * 2 - 1
#         maxx = model.timesteps
#         minn = 0
#         t = torch.empty(1).uniform_(minn, maxx).long()
#         print(t)
#         epsilon = torch.randn(x_0.shape)
        
#         predict = model.unet.forward(model.forward_process(x_0, t), t)
#         criterion = torch.nn.MSELoss()
#         loss = criterion(epsilon , predict)
#         loss.backward()
#         optimizer.step()
#     torch.save(model.state_dict(), 'model_weight.pth')
    
# def get_cifar10_dataloader(batch_size=64):
#     transform = transforms.Compose([
#         transforms.Resize((32, 32)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
#     ])
#     train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     return train_loader


# # 修正後のTraining関数
# def TrainingCIFAR(model, optimizer, num_epochs=100):
#     model.train()
#     device = next(model.parameters()).device
    
#     # CIFAR-10のデータローダーを取得
#     # バッチサイズを小さくするとメモリ使用量を抑えられる（例: 16や32）
#     cifar10_loader = get_cifar10_dataloader(batch_size=16) 

#     for epoch in range(num_epochs):
#         print(f"--- Epoch {epoch + 1}/{num_epochs} ---")
#         # データローダーからバッチ単位で画像を取り出す
#         # CIFAR-10はラベルも返すが、今回は使わないので '_' で受け取る
#         if epoch == 10:
#             break
#         for i, (images, _) in enumerate(cifar10_loader):
#             if i % 30 == 0:
#                 print(f"i = {i}")
#             optimizer.zero_grad()
            
#             # 画像をモデルと同じデバイスに送る
#             x_0 = images.to(device)

#             # タイムステップとノイズも同じデバイス上に作成
#             t = torch.randint(0, model.timesteps, (x_0.shape[0],), device=device).long()
#             epsilon = torch.randn_like(x_0)

#             # ノイズ画像を生成
#             noisy_image = model.forward_process(x_0, t)
#             # ノイズを予測
#             predicted_noise = model.unet.forward(noisy_image, t)
            
#             criterion = torch.nn.MSELoss()
#             loss = criterion(epsilon, predicted_noise)
            
#             loss.backward()
#             optimizer.step()

#             if (i + 1) % 100 == 0:
#                 print(f"Batch [{i+1}/{len(cifar10_loader)}], Loss: {loss.item():.6f}")
#                 break

#     torch.save(model.state_dict(), 'model_weight_cifar10.pth')
#     print("Training finished and model saved.")



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
        model.load_state_dict(torch.load('model_weight_cifar10.pth', map_location=device))
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
        for t in reversed(range(0, model.timesteps)):
            # 現在のタイムステップをモデル入力用のテンソル形式に変換
            timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # 1ステップ分のノイズ除去を実行し、画像を更新
            img = model.reverse_onestep(img, timestep)
    
    print("Image generation complete.")

    # 5. 生成した画像を保存
    # [-1, 1] の範囲で出力される画像を [0, 1] に変換し、PILで扱えるようにCPUに送る
    if torch.isnan(img).any():
        print("NaN detected in generated image!")
    if torch.isinf(img).any():
        print("Inf detected in generated image!")

    generated_image = (img.clamp(-1, 1) + 1) / 2
    
    save_tensor_as_image(generated_image.squeeze(0).cpu(), "generated_image.png")

# def Training_test():
#     model = GaussianDiffusion()
#     optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
#     Training(model, optimizer)


def main():
    model = GaussianDiffusion()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    TrainingFFHQ(model, optimizer)
    #InferTest()
main()