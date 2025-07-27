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
