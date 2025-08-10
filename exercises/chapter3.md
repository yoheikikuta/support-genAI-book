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

---

### 演習問題3.5
最も単純な場合として、$`\mathrm{softmax}`$ 関数 $`y_i = \frac{\exp (x_i)}{ \sum_k \exp (x_k) }`$ の微分を計算し、その振る舞いを見てみる。

```math
\begin{align}
\frac{\partial y_i}{\partial x_j} &=& \frac{ \exp (x_i) \delta_{ij} \sum_k \exp (x_k) - \exp (x_i) \sum_k (\exp (x_k) \delta_{kj} ) }{ (\sum_k \exp (x_k))^2 } \\
&=& \frac{\exp (x_i)}{\sum_k \exp (x_k)} \left( \delta_{ij} - \frac{\exp (x_j)}{\sum_k \exp (x_k)} \right) \\
&=& y_i (\delta_{ij} - y_j)
\end{align}
```

これは、$`i=j`$ のときは $`y_i (1 - y_i)`$ となり、$`y_i \rightarrow 0, 1`$ すなわち $`x_i \rightarrow - \infty, \infty`$ で $`\frac{\partial y_i}{\partial x_i} \rightarrow 0`$ となる。
また、$`i \neq j`$ のときは $`- y_i y_j`$ となり、$`x_{i(j)} \rightarrow - \infty, \infty`$ で $`\frac{\partial y_i}{\partial x_j} \rightarrow 0`$ となる（$`- \infty`$ の要素があればその要素で `y` は $`0`$ になり、$`\infty`$ の要素があれば他の要素で `y` は $`0`$ になる）。
これらは $`\mathrm{softmax}`$ 関数の引数の値（の絶対値）が大きい場合は勾配が小さい領域となり、学習が進みにくくなることが理解できる。

実際の学習では他の関数と組み合わせて使うため、組み合わせによって振る舞いは変わり得る。
例えば、$`\mathrm{softmax}`$ 関数の出力を用いて交差エントロピーを計算する場合 $`L = - \sum_k t_k \log \ y_k`$（$`t_k`$ は正解ラベルの場合で $`1`$ でそれ以外は $`0`$）を考えてみよう。

```math
\begin{align}
\frac{\partial L}{\partial x_i} &=& - \sum_k t_k \frac{1}{y_k} y_k (\delta_{ki} - y_i) \\
&=& y_i - t_i
\end{align}
```

このとき、$`i`$ が正解クラスであれば（$`t_i = 1`$）、$`y_i \rightarrow 1`$（正しい予測）では勾配が小さくなって学習が進みづらくなり、$`y_i \rightarrow 0`$（正しくない予測）では勾配が大きくなって学習が進みやすくなる。
$`i`$ が正解クラスでなければ（$`t_i = 0`$）、$`y_i \rightarrow 1`$（正しくない予測）では勾配が大きくなって学習が進みやすくなり、$`y_i \rightarrow 0`$（正しい予測）では勾配が小さくなって学習が進みにくくなる。
これらは望ましい性質であり、この意味で$`\mathrm{softmax}`$ 関数と交差エントロピーは相性が良いと言える。

-----

### 演習問題3.6
以下に示されている通り、本書の図3.11の結果が得られる。
この最大パス長は膨張畳み込みを使わない通常の畳み込みを用いた（カーネルサイズを $`r`$ とした場合の） CNN と同じになる。

![](/figure/exercise-3-6.png)

-----

### 演習問題3.7
モデル定義は https://github.com/tensorflow/models/blob/61f63bd/official/transformer/model/transformer.py にある。
`Transformer` クラスの `__call__` メソッドは以下であり、モデルの予測対象である `targets` がある場合には学習のために `logits` を計算し、ない場合にはトークン列全体を予測するものになっている。

```python
  def __call__(self, inputs, targets=None):
    """Calculate target logits or inferred target sequences.

    Args:
      inputs: int tensor with shape [batch_size, input_length].
      targets: None or int tensor with shape [batch_size, target_length].

    Returns:
      If targets is defined, then return logits for each word in the target
      sequence. float tensor with shape [batch_size, target_length, vocab_size]
      If target is none, then generate output sequence one token at a time.
        returns a dictionary {
          output: [batch_size, decoded length]
          score: [batch_size, float]}
    """
    # Variance scaling is used here because it seems to work in many problems.
    # Other reasonable initializers may also work just as well.
    initializer = tf.variance_scaling_initializer(
        self.params.initializer_gain, mode="fan_avg", distribution="uniform")
    with tf.variable_scope("Transformer", initializer=initializer):
      # Calculate attention bias for encoder self-attention and decoder
      # multi-headed attention layers.
      attention_bias = model_utils.get_padding_bias(inputs)

      # Run the inputs through the encoder layer to map the symbol
      # representations to continuous representations.
      encoder_outputs = self.encode(inputs, attention_bias)

      # Generate output sequence if targets is None, or return logits if target
      # sequence is known.
      if targets is None:
        return self.predict(encoder_outputs, attention_bias)
      else:
        logits = self.decode(targets, encoder_outputs, attention_bias)
        return logits
```

`targets` がある場合から見ていくと、`decode` というメソッド https://github.com/tensorflow/models/blob/61f63bd/official/transformer/model/transformer.py#L128-L164 が呼ばれている。
本書でも紹介した学習時に用いる shifted right の処理に注目すると、以下のコードで実現されている。
これは、データが `[batch_size, input_length, hidden_size]` という shape であるので、入力系列の最初に1つパディングをして右にずらすとともに、元々の入力列の最後のトークンは除いており、所望の処理になっている。

```python
      with tf.name_scope("shift_targets"):
        # Shift targets to the right, and remove the last element
        decoder_inputs = tf.pad(
            decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
```

`targets` がない場合が推論時の処理であり、`predict` というメソッド https://github.com/tensorflow/models/blob/61f63bd/official/transformer/model/transformer.py#L205-L242 が呼ばれている。
ポイントだけ抜き出してみると、まず以下のように次のトークンを予測するために必要になる `logits` を計算する関数を定義する。

```python
  def _get_symbols_to_logits_fn(self, max_decode_length):
    """Returns a decoding function that calculates logits of the next tokens."""

    timing_signal = model_utils.get_position_encoding(
        max_decode_length + 1, self.params.hidden_size)
    decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
        max_decode_length)

    def symbols_to_logits_fn(ids, i, cache):
      """Generate logits for next potential IDs.

      Args:
        ids: Current decoded sequences.
          int tensor with shape [batch_size * beam_size, i + 1]
        i: Loop index
        cache: dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.

      Returns:
        Tuple of
          (logits with shape [batch_size * beam_size, vocab_size],
           updated cache values)
      """
      # Set decoder input to the last generated IDs
      decoder_input = ids[:, -1:]

      # Preprocess decoder input by getting embeddings and adding timing signal.
      decoder_input = self.embedding_softmax_layer(decoder_input)
      decoder_input += timing_signal[i:i + 1]

      self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
      decoder_outputs = self.decoder_stack(
          decoder_input, cache.get("encoder_outputs"), self_attention_bias,
          cache.get("encoder_decoder_attention_bias"), cache)
      logits = self.embedding_softmax_layer.linear(decoder_outputs)
      logits = tf.squeeze(logits, axis=[1])
      return logits, cache
    return symbols_to_logits_fn
```

`predict` メソッドの全容は以下である。
先程の `logits` を計算する関数をビームサーチに渡すことで、EOS もしくは指定した最大出力長までトークンを順次計算していくことになる（エンコーダーの情報は `cache` に格納されている）。
ビームサーチの実装は興味があれば追ってもらえばよいとして、本書に書かれているように、推論時はトークン $`t_i`$ を計算しなければトークン $`t_{i+1}`$ を計算できないので、1つ先のトークンを（ビームサーチ）で計算してそれに基づいてさらに1つ先のトークンを計算するという逐次処理をしていることが分かる。

```python
  def predict(self, encoder_outputs, encoder_decoder_attention_bias):
    """Return predicted sequence."""
    batch_size = tf.shape(encoder_outputs)[0]
    input_length = tf.shape(encoder_outputs)[1]
    max_decode_length = input_length + self.params.extra_decode_length

    symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length)

    # Create initial set of IDs that will be passed into symbols_to_logits_fn.
    initial_ids = tf.zeros([batch_size], dtype=tf.int32)

    # Create cache storing decoder attention values for each layer.
    cache = {
        "layer_%d" % layer: {
            "k": tf.zeros([batch_size, 0, self.params.hidden_size]),
            "v": tf.zeros([batch_size, 0, self.params.hidden_size]),
        } for layer in range(self.params.num_hidden_layers)}

    # Add encoder output and attention bias to the cache.
    cache["encoder_outputs"] = encoder_outputs
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

    # Use beam search to find the top beam_size sequences and scores.
    decoded_ids, scores = beam_search.sequence_beam_search(
        symbols_to_logits_fn=symbols_to_logits_fn,
        initial_ids=initial_ids,
        initial_cache=cache,
        vocab_size=self.params.vocab_size,
        beam_size=self.params.beam_size,
        alpha=self.params.alpha,
        max_decode_length=max_decode_length,
        eos_id=EOS_ID)

    # Get the top sequence for each batch element
    top_decoded_ids = decoded_ids[:, 0, 1:]
    top_scores = scores[:, 0]

    return {"outputs": top_decoded_ids, "scores": top_scores}
```

-----

### 演習問題3.8
$`\ell_{\mathrm{rate}} = d_{\mathrm{model}}^{-0.5} \ \min (n_{\mathrm{step}}^{-0.5}, n_{\mathrm{step}} n_{\mathrm{warmup\_step}}^{-1.5})`$ であるので、$`\min`$ の中身を見ることで $`n_{\mathrm{step}} = n_{\mathrm{warmup\_step}}`$ で値の大小が変わることが分かる。
$`n_{\mathrm{step}}`$ が $`n_{\mathrm{warmup\_step}}`$ になるまでは $`\min`$ の後者の方が小さくなるので線形に上昇し（warm up の振る舞い）、$`n_{\mathrm{warmup\_step}}`$ を越えると $`-0.5`$ 乗で減衰していく。

プロットは以下である。

![](/figure/exercise-3-8.png)

このプロットを作成するためのコードは以下である。

```python
import matplotlib.pyplot as plt
import numpy as np

d_model = 512
n_warmup = 4000

def lrate(n_step, d_model, n_warmup):
  return (d_model ** (-0.5)) * np.minimum(n_step ** (-0.5), n_step * (n_warmup ** (-1.5)))

x_values = np.linspace(1, 50000, 1000)
y_values = lrate(x_values, d_model, n_warmup)

plt.plot(x_values, y_values)
plt.xlabel('n_step')
plt.ylabel('lrate')
plt.title(f'Plot of lrate')
plt.grid(True)
plt.show()
```

