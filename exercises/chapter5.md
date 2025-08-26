# 第5章 演習問題の解答

### 演習問題5.1
原論文 An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale [https://arxiv.org/abs/2010.11929v2 ](https://arxiv.org/abs/2010.11929v2) の 4.1 SETUP 内の Metrics. に以下のような記述がある。

> We report results on downstream datasets either through few-shot or fine-tuning accuracy. Fine-tuning accuracies capture the performance of each model after fine-tuning it on the respective dataset. Few-shot accuracies are obtained by solving a regularized least-squares regression problem that maps the (frozen) representation of a subset of training images to $`\{−1, 1\}^K`$ target vectors. This formulation allows us to recover the exact solution in closed form. Though we mainly focus on fine-tuning performance, we sometimes use linear few-shot accuracies for fast on-the-fly evaluation where fine-tuning would be too costly.

ここに直接的に記述されているように、少数ショットの場合に線形分類器を正則化された最小二乗回帰で構築しているのは、（よく知られているように）閉形式で解を求めることができ、正確かつ高速に処理ができるためである。
最も大きな恩恵は計算資源の節約であり、原論文でも別の箇所で `To save compute` と言及されている。

事前学習済みのモデルを特定のタスクに適用するためにはファインチューニングが典型的な手法であるが、扱うモデルのサイズが大きい・比較するモデルの数が多い・対象とするタスクが多い、などの場合には実験の計算コストが高くなる。
この計算コストを抑えたい場合に、ファインチューニングではなく線形分類器を正則化された最小二乗回帰で構築している。

---

### 演習問題5.2
原論文 An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale [https://arxiv.org/abs/2010.11929v2 ](https://arxiv.org/abs/2010.11929v2) の Appendix D.8 ATTENTION MAPS に以下のような記述がある。

> To compute maps of the attention from the output token to the input space (Figures 6 and 14), we used Attention Rollout (Abnar & Zuidema, 2020). Briefly, we averaged attention weights of ViT-L/16 across all heads and then recursively multiplied the weight matrices of all layers. This accounts for the mixing of attention across tokens through all layers.

ここで登場する Attention Rollout の論文は Quantifying Attention Flow in Transformers [https://arxiv.org/abs/2005.00928v2](https://arxiv.org/abs/2005.00928v2) である。
この論文の動機は層を重ねた際の注意重みの情報の流れを定量化である。
以下の図はこの論文の Figure 1 の引用で、入力として動詞の位置までの文が与えられたときにその動詞が単数か複数かを予測する問題を対象に注意重みを可視化しており、(a)が生の注意重みで(b)と(c)が提案手法である。
(a)を見ると、層を重ねると注意重みが一様になっており、ここだけを見ても入力トークンの相対的重要度に対応していないため、入力層まで遡って注意重みを考慮することでモデルに関するより効果的な洞察を得ようとして提案手法を考案している。
公式実装は [https://github.com/samiraabnar/attention_flow/tree/master](https://github.com/samiraabnar/attention_flow/tree/master) で与えられている。

![](/figure/exercise-5-2-1.png)

ここでは Attention Rollout のみに注目する。
これは埋め込みに対する注意を入力として与えたとき、与えられたモデルの各層における注意を再帰的に計算する手法で、入力層から注意の情報がどのように渡っていくのかを考慮したものである。

一つ考慮しないといけない点は、残差接続の存在である。
残差接続でも情報が伝播していくため、残差接続も考慮して注意の情報を処理していく必要がある。
具体的には、残差接続がある場合には $`\boldsymbol{h}_{\ell + 1} = (W_{\mathrm{att}} + I) \boldsymbol{h}_{\ell}`$ と単位行列を足してそのあと再度正規化した上でこれを生の注意行列 $`A`$ と呼び、各層の情報を繋いでいく。

層 $`\ell_i, \ell_j`$（これは単に層のインデックス）の生の注意行列を$`A(\ell_i), A(\ell_j)`$ として、以下のように再帰的に掛けて得られる注意行列を $`\tilde{A}(\ell_i)`$ とする。

```math
\begin{align}
\tilde{A}(\ell_i)=
\begin{cases}
A(\ell_i)\,\tilde{A}(\ell_{i-1}) & \text{if} \ i>j,\\
A(\ell_i) & \text{if} \ i=j.
\end{cases}
\end{align}
```

これが Attention Rollout であり、典型的には最終層の特定のトークン位置に対して他のトークン位置がどのように寄与しているかを可視化する道具となる。
この論文はテキストに対して分析しているが、画像においても同様で、例えば以下のような結果が得られる。

![](/figure/exercise-5-2-2.png)

これの Colab 実装は [https://colab.research.google.com/drive/1jKAexuO3bOgi4znlHOawcv85sQ4LU7bP?usp=sharing](https://colab.research.google.com/drive/1jKAexuO3bOgi4znlHOawcv85sQ4LU7bP?usp=sharing)

---

### 演習問題5.3
Chapman-Kolmogorov 方程式は以下で与えられる。

```math
\begin{equation}
  u(r,x;t,z) = \int^\infty_{-\infty} dy \ u(r,x;s,y) u(s,y;t,z)
\end{equation}
```

後ろ向き方程式の導出は $`t,z`$ を固定して、$`r,x`$ を変数と見て、前向き方程式と同様の手順を踏めばよい。
偏微分の定義式

```math
\begin{equation}
  \frac{\partial u (r,x;t,z)}{\partial r} = \lim_{dr \rightarrow 0} \frac{1}{dr} \left( u(r,x;t,z) - u(r-dr,x;t,z) \right)
\end{equation}
```

に Chapman-Kolmogorov 方程式を代入すると以下が得られる。

```math
\begin{align}
  \frac{\partial u (r,x;t,z)}{\partial r} &= \lim_{dr \rightarrow 0} \frac{1}{dr} \left[ u(r,x;t,z) - \int^\infty_{-\infty} dy \ u(r-dr,x;r,y) u(r,y;t,z) \right]
\end{align}
```

括弧内の第2項を整理する。
微小量 $`\xi = y - x`$ を導入してこれで変数変換をして、本書の式 (5.9) と式 (5.10) を使うことに念頭において式変形をする。

```math
\begin{align}
  \int^\infty_{-\infty} dy \ u(r-dr,x;r,y) u(r,y;t,z) &= \int^\infty_{-\infty} d\xi \ u(r-dr,x;r,x+\xi) u(r,x+\xi;t,z) \\
  &= \int^\infty_{-\infty} d\xi \ u(r-dr,x;r,x+\xi) \left( e^{\xi \frac{\partial}{\partial x}} u(r,x;t,z) \right) \\
  &= \int^\infty_{-\infty} d\xi \ u(r-dr,x;r,x+\xi) \left( \left( 1 + \xi \frac{\partial}{\partial x} + \frac{\xi^2}{2} \frac{\partial^2}{\partial x^2} \right) u(r,x;t,z) + \mathcal{O} (\xi^3) \right)
\end{align}
```

偏微分の定義式 Chapman-Kolmogorov 方程式を代入したものに戻って $`\mathcal{O} (\xi^3)`$ を無視して項を整理し、本書の式 (5.9) と式 (5.10) を使う。

```math
\begin{align}
  \frac{\partial u (r,x;t,z)}{\partial r} &= \lim_{dr \rightarrow 0} \frac{1}{dr} \left[ - \frac{\partial u}{\partial x} \int^\infty_{-\infty} d\xi \ \xi u(r-dr,x;r,x+\xi) - \frac{1}{2} \frac{\partial^2 u}{\partial x^2} \int^\infty_{-\infty} d\xi \ \xi^2 u(r-dr,x;r,x+\xi) \right] \\
  &= - b(r,x) \frac{\partial u(r,x;t,z)}{\partial x} - \frac{a(r,x)}{2} \frac{\partial^2 u(r,x;t,z)}{\partial x^2}
\end{align}
```

これで後ろ向き方程式が導出できた。

---

### 演習問題5.4
本書における式 (5.27) における $`L_{t-1}`$ を、式 (5.28) と分散共分散行列を簡単化して対角化成分のみを考える $`p_\theta (\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t) = \mathcal{N} (\boldsymbol{x}_{t-1}; \boldsymbol{\mu}_\theta (\boldsymbol{x}_t, t), \sigma^2_t I)`$ を用いて、素直に計算すればよい。

```math
\begin{align}
  L_{t-1} &= \mathbb{E}_{q (\boldsymbol{x}_{0:T})} \left[ D_{\mathrm{KL}}\left(q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0) \mid p_\theta(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t)\right) \right] \\
  &= \mathbb{E}_{q (\boldsymbol{x}_{0:T})} \left[ D_{\mathrm{KL}} \left( \mathcal{N} (\boldsymbol{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t (\boldsymbol{x}_t, t), \tilde{\beta}_t I) \mid \mathcal{N} (\boldsymbol{x}_{t-1}; \boldsymbol{\mu}_\theta (\boldsymbol{x}_t, t), \sigma^2_t I) \right) \right] \\
  &= \mathbb{E}_{q (\boldsymbol{x}_{0:T})} \left[ \frac{1}{2 \sigma_t^2} \| \tilde{\boldsymbol{\mu}}_t (\boldsymbol{x}_t, t) -  \boldsymbol{\mu}_\theta (\boldsymbol{x}_t, t) \|^2 \right] + (\mathrm{const})
\end{align}
```

ここで、多変量正規分布同士の KL ダイバージェンスの結果はよく知られている（例えば [https://statproofbook.github.io/P/mvn-kl.html](https://statproofbook.github.io/P/mvn-kl.html) の式 (2) で、その証明もこのページに書かれている）ので、その結果を使って $`\boldsymbol{x}`$ に依存する項のみを残している。

この演習問題は多変量正規分布同士の KL ダイバージェンスの計算が主題でその内容と証明は上記リンクに譲ってしまったが、人生で一度は自分の手で真面目に計算してみるとよい類の計算である。

---

### 演習問題5.5
まず、本書の式 (5.34) をある画素 $`i`$ に注目する。

```math
\begin{align}
  p_{\theta} (x_{0,i} \mid \boldsymbol{x}_1) = \int_{\delta_{-}(x_{0,i})}^{\,\delta_{+}(x_{0,i})} \mathcal{N} (x; \mu_{\theta, i} (\boldsymbol{x}_1, 1), \sigma_1^2) dx
\end{align}
```

境界以外では区間幅が $`\Delta = \delta_{+}(x_{0,i}) - \delta_{-}(x_{0,i}) = \frac{2}{255}`$（式 (5.35) を使用）になるので、これが十分に小さいとして以下のように近似する。

```math
\begin{align}
  p_{\theta} (x_{0,i} \mid \boldsymbol{x}_1) &= \int_{x_{0,i} - \frac{\Delta}{2}}^{x_{0,i} + \frac{\Delta}{2}} \mathcal{N} (x; \mu_{\theta, i} (\boldsymbol{x}_1, 1), \sigma_1^2) dx 
  \simeq \mathcal{N} (x_{0,i}; \mu_{\theta, i} (\boldsymbol{x}_1, 1), \sigma_1^2) \Delta
\end{align}
```

負の対数尤度を取ると、$`\theta`$ に依らない定数部分を $`(\mathrm{const})`$ として以下が得られる。

```math
\begin{align}
  - \log p_{\theta} (x_{0,i} \mid \boldsymbol{x}_1) \simeq \frac{ (x_{0,i} - \mu_{\theta, i} (\boldsymbol{x}_1, 1))^2 }{2 \sigma_1^2} + (\mathrm{const})
\end{align}
```

全ての画素に対して和を取り、境界の効果は無視すると以下が得られる。

```math
\begin{align}
  L_0 \simeq \frac{1}{2 \sigma_1^2} \| \boldsymbol{x}_{0} - \boldsymbol{\mu}_{\theta} (\boldsymbol{x}_1, 1) \|^2 + (\mathrm{const})
\end{align}
```

次にこれを $`\boldsymbol{\varepsilon}`$ で表そう。
式 (5.32) に $`t=1`$ を代入し、$`\bar{\alpha}_1 = \alpha_1, 1 - \bar{\alpha}_1 = \beta_1`$ であることに注意すると以下が得られる。

```math
\begin{align}
  \boldsymbol{\mu}_\theta (\boldsymbol{x}_1, 1) = \frac{1}{\sqrt{\alpha_1}} \left( \boldsymbol{x}_1 - \sqrt{\beta_1} \boldsymbol{\varepsilon}_\theta (\boldsymbol{x}_1, 1) \right)
\end{align}
```

ここで、式 (5.25) に $`t=1`$ を代入した $`\boldsymbol{x}_1 = \sqrt{\alpha_1} \boldsymbol{x}_0 + \sqrt{\beta_1} \boldsymbol{z} \ \text{where} \ \boldsymbol{z} \sim \mathcal{N} (\boldsymbol{0}, I)`$ を使うことで、以下が得られる。

```math
\begin{align}
  \boldsymbol{\mu}_\theta (\boldsymbol{x}_1, 1) = \boldsymbol{x}_0 + \frac{\sqrt{\beta_1}}{\sqrt{\alpha_1}} \left( \boldsymbol{z} - \boldsymbol{\varepsilon}_\theta (\boldsymbol{x}_1, 1) \right)
\end{align}
```

$`\boldsymbol{z}`$ を $`\boldsymbol{\varepsilon}`$ と置き換え、$`L_0`$ の式に代入すると、最終的に以下の形になることが分かる。

```math
\begin{align}
  L_0 \simeq \frac{\beta_1}{2 \sigma_1^2 \alpha_1} \| \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta (\boldsymbol{x}_1, 1) \|^2 + (\mathrm{const})
\end{align}
```

これは係数と定数項を無視すれば式 (5.36) の $`t=1`$ の場合に等しい。

---

### 演習問題5.6
無視した重みは $`\frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)}`$ で、$`\sigma_t^2 = \beta_t`$ とすると $`\frac{\beta_t}{2 \alpha_t (1 - \bar{\alpha}_t)}`$ となり、あとはタイムステップ全体を $`T = 1000`$ として $`\beta_1 = 10^{-4}`$ から $`\beta_T = 0.02`$ まで線形に変化させてプロットすればよい。

プロットした結果は以下で、$`t`$ が小さい場合の重みが相対的にかなり大きくなっていることが分かる。

![](/figure/exercise-5-6.png)

このプロットを作成するための Python コードは以下である。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

T = 1000
beta_1 = 1e-4
beta_T = 0.02

betas = np.linspace(beta_1, beta_T, T, dtype=np.float64)
alphas = 1.0 - betas
bar_alphas = np.cumprod(alphas)

weights = betas / (2.0 * alphas * (1.0 - bar_alphas))
timestep = np.arange(1, T + 1)

plt.figure(figsize=(8, 4.5))
plt.plot(timestep, weights)
plt.xlabel("timestep")
plt.ylabel("Ignored weight")
plt.title("Ignored weight vs timestep t (T=1000, linear betas, $\\sigma_t^2=\\beta_t$)")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
```

---

### 演習問題5.7
公式実装 [https://github.com/facebookresearch/DiT/blob/ed81ce2/models.py](https://github.com/facebookresearch/DiT/blob/ed81ce2/models.py) のうち、入力に関わる部分を読み解いていこう。

モデルの初期化部分は以下のようになっており、本書における Noised Latent の埋め込みが `self.x_embedder = PatchEmbed` で、Timestep の埋め込み $`\boldsymbol{e}_t`$ が `self.t_embedder = TimestepEmbedder` で、条件付けの情報として Label の埋め込み $`\boldsymbol{e}_y`$ が `self.y_embedder = LabelEmbedder` である。

```python
class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
```

`PatchEmbed` は別の repository [https://github.com/huggingface/pytorch-image-models/blob/954613a/timm/layers/patch_embed.py#L26-L137](https://github.com/huggingface/pytorch-image-models/blob/954613a/timm/layers/patch_embed.py#L26-L137) で定義されているもので、2 次元の画像をパッチ化して埋め込みをするものである。

`TimestepEmbedder` の定義は以下で、Transformer の原論文と本質的に同じ $`\mathrm{sin}, \mathrm{cos}`$ による埋め込みに加えて、MLP も適用して特徴量化している。

```python
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
```

`LabelEmbedder` の定義は以下で、 クラス毎に埋め込みを学習するようになっており、分類器なしガイダンスのための dropout も含まれていることが分かる。

```python
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
```

これらの入力は以下のように扱われている。
本書でも解説した通り、Noised Latent の埋め込みは位置埋め込みと足し合わされ、Timestep の埋め込みと Label の埋め込みは足し合わせたものと `block(x, c)` で組み合わされるようになっている。

```python
    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x
```

この `block` は以下で定義される DiT Block であり、本書で解説した通り modulate を使って条件付けの情報を取り込む adaLN にスケールパラメーターも導入した adaLN-Zero を用いている。

```python
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
```
