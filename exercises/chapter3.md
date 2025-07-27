# 第3章 演習問題の解答

### 演習問題3.1
$`\boldsymbol{\mathrm{PE}}_{p+k}`$ が $`\boldsymbol{\mathrm{PE}}_{p}`$ と $`\boldsymbol{\mathrm{PE}}_{k}`$ の線型結合で書けることは以下のように示される。

```math
\begin{align}
\mathrm{PE}_{p+k, 2i} &=& \sin \left( \frac{p}{10000^{2i / d_{\mathrm{model}}}} \right) \cos \left( \frac{k}{10000^{2i / d_{\mathrm{model}}}} \right) + \cos \left( \frac{p}{10000^{2i / d_{\mathrm{model}}}} \right) \sin \left( \frac{k}{10000^{2i / d_{\mathrm{model}}}} \right) = \mathrm{PE}_{p, 2i} \mathrm{PE}_{k, 2i+1} + \mathrm{PE}_{p, 2i+1} \mathrm{PE}_{k, 2i} \\
\mathrm{PE}_{p+k, 2i+1} &=& \mathrm{PE}_{p, 2i+1} \mathrm{PE}_{k, 2i+1} - \mathrm{PE}_{p, 2i} \mathrm{PE}_{k, 2i}
\end{align}
```

これは次のように $`sin`$ と $`cos`$ をペアとすると、回転行列を使って書ける。

```math
\begin{align}
\begin{pmatrix}
   \mathrm{PE}_{p+k, 2i}  \\
   \mathrm{PE}_{p+k, 2i+1} 
\end{pmatrix}
=
\begin{pmatrix}
   \cos \left( \frac{k}{10000^{2i / d_{\mathrm{model}}}} \right) & \sin \left( \frac{k}{10000^{2i / d_{\mathrm{model}}}} \right) \\
   - \sin \left( \frac{k}{10000^{2i / d_{\mathrm{model}}}} \right) & \cos \left( \frac{k}{10000^{2i / d_{\mathrm{model}}}} \right)
\end{pmatrix}
\begin{pmatrix}
   \mathrm{PE}_{p, 2i}  \\
   \mathrm{PE}_{p, 2i+1} 
\end{pmatrix}
\end{align}
```

この結果から、相対的な位置の違い $`k`$ を学習不要で線形変換で表現できるのでモデルが相対的な特徴を簡単に作れること、$`\boldsymbol{\mathrm{PE}}_{p+k}`$ と $`\boldsymbol{\mathrm{PE}}_{p}`$ の内積は相対的な距離 $`k`$ だけに依存すること、が分かる。
この意味で相対的な位置関係を把握しやすい。

---

### 演習問題3.2
colab で実装した例: https://colab.research.google.com/drive/1bMMEM0lPgqfxFeOGvZI-uYrb0u-BcDIw?usp=sharing

![](/figure/exercise-3-2.png)

---

### 演習問題3.3
colab で実装した例: https://colab.research.google.com/drive/1ECLiKlBOo7kEsoSFddG0tWfRkrpJ5ZB6?usp=sharing

---

### 演習問題3.4
本書の定義に沿って計算を進めていく。まず、 $`Q K^{\mathsf{T}}`$ は以下の $`n`$ 行 $`n`$ 列の行列である。

```math
\begin{align}
Q K^{\mathsf{T}} = 
\begin{pmatrix}
   \boldsymbol{h}^{\mathrm{query} \mathsf{T}}_1  \\
   \vdots  \\
   \boldsymbol{h}^{\mathrm{query} \mathsf{T}}_n
\end{pmatrix}
\begin{pmatrix}
   \boldsymbol{h}^{\mathrm{key}}_1 & \cdots & \boldsymbol{h}^{\mathrm{key}}_n
\end{pmatrix}
=
\begin{pmatrix}
   \boldsymbol{h}^{\mathrm{query}}_1 \cdot \boldsymbol{h}^{\mathrm{key}}_1 & \cdots & \boldsymbol{h}^{\mathrm{query}}_1 \cdot \boldsymbol{h}^{\mathrm{key}}_n \\
   \vdots & \vdots & \vdots \\
   \boldsymbol{h}^{\mathrm{query}}_n \cdot \boldsymbol{h}^{\mathrm{key}}_1 & \cdots & \boldsymbol{h}^{\mathrm{query}}_n \cdot \boldsymbol{h}^{\mathrm{key}}_n
\end{pmatrix}
\end{align}
```

次に、$`\mathrm{softmax}`$ が行方向に適用されることに注意すると、以下が得られる。

```math
\begin{align}
\mathrm{softmax} (Q K^{\mathsf{T}})
=
\begin{pmatrix}
   \frac{ \exp(\boldsymbol{h}^{\mathrm{query}}_1 \cdot \boldsymbol{h}^{\mathrm{key}}_1) }{ \sum_j^n \exp(\boldsymbol{h}^{\mathrm{query}}_1 \cdot \boldsymbol{h}^{\mathrm{key}}_j) } & \cdots & \frac{ \exp(\boldsymbol{h}^{\mathrm{query}}_1 \cdot \boldsymbol{h}^{\mathrm{key}}_n) }{ \sum_j^n \exp(\boldsymbol{h}^{\mathrm{query}}_1 \cdot \boldsymbol{h}^{\mathrm{key}}_j) } \\
   \vdots & \vdots & \vdots \\
   \frac{ \exp(\boldsymbol{h}^{\mathrm{query}}_n \cdot \boldsymbol{h}^{\mathrm{key}}_1) }{ \sum_j^n \exp(\boldsymbol{h}^{\mathrm{query}}_n \cdot \boldsymbol{h}^{\mathrm{key}}_j) } & \cdots & \frac{ \exp(\boldsymbol{h}^{\mathrm{query}}_n \cdot \boldsymbol{h}^{\mathrm{key}}_n) }{ \sum_j^n \exp(\boldsymbol{h}^{\mathrm{query}}_n \cdot \boldsymbol{h}^{\mathrm{key}}_j) }
\end{pmatrix}
\end{align}
```

これに $`V`$ を右から掛ければ、以下の $`n`$ 行 $`d_{\mathrm{value}}`$ 列の行列が得られる。

```math
\begin{align}
\mathrm{softmax} (Q K^{\mathsf{T}}) V
=
\begin{pmatrix}
   \sum_t^n \frac{ \exp(\boldsymbol{h}^{\mathrm{query}}_1 \cdot \boldsymbol{h}^{\mathrm{key}}_t) }{ \sum_j^n \exp(\boldsymbol{h}^{\mathrm{query}}_1 \cdot \boldsymbol{h}^{\mathrm{key}}_j) } \boldsymbol{h}^{\mathrm{value}}_t \\
   \vdots  \\
   \sum_t^n \frac{ \exp(\boldsymbol{h}^{\mathrm{query}}_n \cdot \boldsymbol{h}^{\mathrm{key}}_t) }{ \sum_j^n \exp(\boldsymbol{h}^{\mathrm{query}}_n \cdot \boldsymbol{h}^{\mathrm{key}}_j) } \boldsymbol{h}^{\mathrm{value}}_t
\end{pmatrix}
\end{align}
```

ここで、 $`k`$ 行目を抜き出して、$`a_{k,t} = \frac{ \exp(\boldsymbol{h}^{\mathrm{query}}_k \cdot \boldsymbol{h}^{\mathrm{key}}_t) }{ \sum_j^n \exp(\boldsymbol{h}^{\mathrm{query}}_k \cdot \boldsymbol{h}^{\mathrm{key}}_j) }`$ とすれば、以下が得られるので、各行（トークンの位置）ごとにクエリーとキーのスコアで重み付けしたバリューを計算しており、確かに注意機構で実現したい計算になっていることが分かる。

```math
\begin{align}
\mathrm{softmax} (Q K^{\mathsf{T}}) V
=
\begin{pmatrix}
   \sum_t^n a_{1,t} \boldsymbol{h}^{\mathrm{value}}_t \\
   \vdots  \\
   \sum_t^n a_{n,t} \boldsymbol{h}^{\mathrm{value}}_t 
\end{pmatrix}
\end{align}
```
