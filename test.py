import torch
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

# --- 使用例 ---
batch_size = 4
embedding_dim = 128

# 時間埋め込みモジュールを初期化
time_embed_module = SinusoidalPositionEmbeddings(dim=embedding_dim)

# ランダムなタイムステップのバッチを作成 (実際の値は0から1000など)
timesteps = torch.randint(0, 1000, (batch_size,))
print(f"timesteps = {timesteps}")
# 時間埋め込みを計算
time_embeddings = time_embed_module(timesteps)

print("Input timesteps shape:", timesteps.shape)
print("Output embeddings shape:", time_embeddings.shape)

# >> Input timesteps shape: torch.Size([4])
# >> Output embeddings shape: torch.Size([4, 128])