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
    img =  load_image_as_tensor("./a.png").unsqueeze(0).to(device)
    img = 2*img - 1
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

    #generated_image = (img2.clamp(-1, 1) + 1) / 2
    
    save_tensor_as_image(img2.squeeze(0).cpu(), "generated_image.png")




def main():
    print("code execute!!")
    # model = GaussianDiffusion()
    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    #Training_test()
    InferTest()
main()