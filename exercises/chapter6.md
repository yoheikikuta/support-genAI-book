# 第6章 演習問題の解答

### 演習問題6.1
バッチの次元を $`n`$ として、画像データを $`I \in \mathbb{R}^{n \times hwc}`$、テキストデータを $`T \in \mathbb{R}^{n \times \ell}`$ とする。
これらのデータに画像エンコーダー $`f_i: \mathbb{R}^{n \times hwc} \rightarrow \mathbb{R}^{n \times d_i}`$ とテキストエンコーダー $`f_t: \mathbb{R}^{n \times \ell} \rightarrow \mathbb{R}^{n \times d_t}`$ を適用したものを $`I_f = f_i (I) \in \mathbb{R}^{n \times d_i}, T_f = f_t (T) \in \mathbb{R}^{n \times d_t}`$ とする。

これらはエンコーダーから得られる特徴量であり、一般に画像とテキストで特徴量次元が異なるため、次元を合わせて同じ空間で扱えるようにするために変換行列 $`W_i \in \mathbb{R}^{d_i \times d_e}, W_t \in \mathbb{R}^{d_t \times d_e}`$ を導入し、値の大きさも同程度になるように正規化も施す。

```math
\begin{align}
I_e &=& \mathrm{norm} (I_f W_i) \in \mathbb{R}^{n \times d_e} \\
T_e &=& \mathrm{norm} (T_f W_t) \in \mathbb{R}^{n \times d_e}
\end{align}
```

ただし、$`\mathrm{norm}`$ は各行に対して適用する関数である。
このような処理は数式で表現しようとすると不要に煩雑になりがちなので自然言語で表現してしまうことも多く、プログラムの方が処理が簡潔に書ける（そのような処理をする関数をみんながよく使うので共通認識が出来上がっている）。

続いて、これらの行列の積を取って各データの特徴量同士のコサイン類似度を計算し、スケール因子も導入する（これは後でクロスエントロピー損失を計算する際に効いてくる）。

```math
\begin{align}
\mathrm{logits} = ( I_e T_e^{\mathsf{T}} ) e^t \in \mathbb{R}^{n \times n}
\end{align}
```

$`\mathrm{logits}`$ は行を固定するとある画像に対する各テキストのコサイン類似度にスケール因子を乗じたものになっており、列を固定するとその逆である。
また、対角成分が正しい組み合わせ（ある画像とその画像に対して付与されているキャプションテキスト）となっていることに注意されたい。

これを使って行方向と列方向それぞれでクロスエントロピー損失を計算する。
例えば行方向のクロスエントロピー損失 $`\mathrm{loss}_i`$ は、$`a`$ 番目の行に注目して正解ラベルが対角成分であることに注意すると、以下のように書ける。

```math
\begin{align}
\mathrm{loss}_{i,a} = - \sum_b^n \delta_{ab} \log p_{ab} = - \log p_{aa} \ \ \text{where} \ \ p_{ab} = \frac{ \exp (\mathrm{logits}_{ab}) }{ \sum_c^n \exp (\mathrm{logits}_{ac}) }
\end{align}
```

これを全ての行で平均を取ることで $`\mathrm{loss}_i`$ が得られる。

```math
\begin{align}
\mathrm{loss}_{i} = - \frac{1}{n} \sum_a^n \log p_{aa} 
\end{align}
```

データの作り方から対角成分だけが残るようになっているが、この対角成分の確率 $`p_{aa}`$ は画像を固定して各テキストとの類似度を計算して規格化して得られたものであることに注意されたい。
同様のことを列方向で実施することで、クロスエントロピー損失 $`\mathrm{loss}_t`$ が得られる。

```math
\begin{align}
\mathrm{loss}_{t} = - \frac{1}{n} \sum_a^n \log p'_{aa}  \ \ \text{where} \ \ p'_{ab} = \frac{ \exp (\mathrm{logits}_{ab}) }{ \sum_c^n \exp (\mathrm{logits}_{cb}) }
\end{align}
```

最終的に、2 つの損失を平均することで最終的な損失が得られる。

```math
\begin{align}
\mathrm{loss} = \frac{\mathrm{loss}_i + \mathrm{loss}_t}{2}
\end{align}
```


この擬似コードはかなり丁寧にデータの shape も記載してあるので数式に翻訳する難易度は低いものだったが、ものによっては情報量が少ない場合もあり、数式に翻訳するのは理解を深めるための良いきっかけになる。

---

### 演習問題6.2
`model.py` [https://github.com/mlfoundations/open_clip/blob/63fbfd8/src/open_clip/model.py](https://github.com/mlfoundations/open_clip/blob/63fbfd8/src/open_clip/model.py) に書かれている。
2 つのクラス `CLIP` と `CustomTextCLIP` があり、前者は OpenAI の CLIP 実装 [https://github.com/openai/CLIP/tree/main](https://github.com/openai/CLIP/tree/main) の実装と重みを使うことを念頭に置いたもので、後者は抽象化して必要に応じて HuggingFace や独自モデルを使えるように拡張してある。

ここでは、後者の `CustomTextCLIP` の実装を読んでみる。
画像とテキストの特徴量はそれぞれ `encode_image` と `encode_text` という method で取得しているが、これらの特徴量を計算するエンコーダーは `_build_vision_tower` と `_build_text_tower` で与えられている。

`_build_vision_tower` の実装は以下の通りで、具体的には次のモデルが利用可能である。

- timm [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) のモデル名が与えられればそれを使用
- tuple か list なら `ModifiedResNet` を使用（ResNet は各ステージでいくつの残差ブロックを積むかを tuple や list で与えるのが慣例）
- それ以外なら `VisionTransformer` を使用

```python
def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if vision_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)

        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual
```

`_build_text_tower` の実装は以下の通りで、具体的には次のモデルが利用可能である。

- HuggingFace のモデル名が与えられればそれを使用（HuggingFace のモデルを扱うためのアダプターの実装も準備している [https://github.com/mlfoundations/open_clip/blob/63fbfd8/src/open_clip/hf_model.py#L96-L193](https://github.com/mlfoundations/open_clip/blob/63fbfd8/src/open_clip/hf_model.py#L96-L193) ）
- それ以外なら `TextTransformer` を使用

```python
def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj_type=text_cfg.hf_proj_type,
            pooler_type=text_cfg.hf_pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if text_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
        if text_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **text_cfg.act_kwargs)

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            mlp_ratio=text_cfg.mlp_ratio,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            no_causal_mask=text_cfg.no_causal_mask,
            pad_id=text_cfg.pad_id,
            pool_type=text_cfg.pool_type,
            proj_type=text_cfg.proj_type,
            proj_bias=text_cfg.proj_bias,
            output_tokens=text_cfg.output_tokens,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text
```

再現実装では拡張性を高めて timm や HuggingFace のモデルが使えるようになっているが、これら以外の具体的なモデル実装である、画像なら `ModifiedResNet` と `VisionTransformer` でテキストなら `TextTransformer` がオリジナルの CLIP でも使われている具体的なモデルである。

---

### 演習問題6.3
まず、2 個の確率変数 $`X,Y`$ の条件付き確率の定義は、$`p(Y) > 0`$ として次で与えられる。

```math
p (X \mid Y) = \frac{p (X, Y)}{p(Y)}
```

これに対して新たに条件 $`Z`$ を課すことを考えると、定義から進めて以下が得られる。

```math
p (X, Y \mid Z) = \frac{p(X,Y,Z)}{p(Z)} = \frac{ p(X \mid Y, Z) p(Y, Z) }{p(Z)} =  p (X \mid Y, Z) p(Y \mid Z)
```

これが 3 個の確率変数 $`X,Y,Z`$ の条件付き確率の連鎖律である。
あとは本書の表記と合わせるために、$`X \rightarrow \boldsymbol{x}, Y \rightarrow \boldsymbol{z}_x, Z \rightarrow \boldsymbol{y}`$ と読み替えれば、本書の記述に従い式 (6.1) が導出できる。

---

### 演習問題6.4
再現実装 [https://github.com/lucidrains/DALLE2-pytorch](https://github.com/lucidrains/DALLE2-pytorch) を読み解いていけばいい。
1 ファイル [https://github.com/lucidrains/DALLE2-pytorch/blob/680dfc4/dalle2_pytorch/dalle2_pytorch.py](https://github.com/lucidrains/DALLE2-pytorch/blob/680dfc4/dalle2_pytorch/dalle2_pytorch.py) に大量の情報が詰め込まれていて読みづらいが、必要な箇所だけを拾い読みする。

事前モデルの具体的な実装は `DiffusionPriorNetwork` クラスであり、`forward` メソッドの最後の部分は以下のようになっている。
`tokens` は本書における入力 $`\boldsymbol{y}, \boldsymbol{z}_y, \boldsymbol{e}_t, \boldsymbol{z}_{x,t}`$ に対応していて、最後の `learned_queries` はノイズを除去した CLIP 画像特徴量の予測を取り出すための入力である（これは実装上有用）。

```python
    def forward(
        self,
        image_embed,
        diffusion_timesteps,
        *,
        text_embed,
        text_encodings = None,
        self_cond = None,
        text_cond_drop_prob = 0.,
        image_cond_drop_prob = 0.
    ):

   # 途中の部分は省略

        tokens = torch.cat((
            text_encodings,
            text_embed,
            time_embed,
            image_embed,
            learned_queries
        ), dim = -2)

        # attend

        tokens = self.causal_transformer(tokens)

        # get learned query, which should predict the image embedding (per DDPM timestep)

        pred_image_embed = tokens[..., -1, :]

        return pred_image_embed
```

問題となっているのは $`\boldsymbol{y}, \boldsymbol{z}_y`$ が具体的にどのようなものになっているかで、これは `text_encodings, text_embed` に対応しているので、これらを調べればよい。
少しコードを読むと、これらの変数はテキスト特徴量を得るためにどの Adapter を使うかで変わるものだが、ここではその中の一つとして `OpenAIClipAdapter(BaseClipAdapter)` を取り上げよう。
`text_embed` は `clip` のテキスト出力で文章全体の情報を含む 1 つの特徴ベクトルであり、`text_encodings` は `ln_final` 層にフックして各トークン毎の最終層の特徴ベクトルを取得している。

```python
class OpenAIClipAdapter(BaseClipAdapter):
    def __init__(
        self,
        name = 'ViT-B/32'
    ):
        import clip
        openai_clip, preprocess = clip.load(name)
        super().__init__(openai_clip)
        self.eos_id = 49407 # for handling 0 being also '!'

        text_attention_final = self.find_layer('ln_final')

        self.dim_latent_ = text_attention_final.weight.shape[0]
        self.handle = text_attention_final.register_forward_hook(self._hook)

        self.clip_normalize = preprocess.transforms[-1]
        self.cleared = False

    def clear(self):
        if self.cleared:
            return

        self.handle()

    def _hook(self, _, inputs, outputs):
        self.text_encodings = outputs

    # 途中省略

    @torch.no_grad()
    def embed_text(self, text):
        text = text[..., :self.max_text_len]

        is_eos_id = (text == self.eos_id)
        text_mask_excluding_eos = is_eos_id.cumsum(dim = -1) == 0
        text_mask = F.pad(text_mask_excluding_eos, (1, -1), value = True)
        text_mask = text_mask & (text != 0)
        assert not self.cleared

        text_embed = self.clip.encode_text(text)
        text_encodings = self.text_encodings
        text_encodings = text_encodings.masked_fill(~text_mask[..., None], 0.)
        del self.text_encodings
        return EmbeddedText(l2norm(text_embed.float()), text_encodings.float())
```

再現実装ではあるが、これで $`\boldsymbol{y}, \boldsymbol{z}_y`$ として具体的に何が使われているかの例を理解できた。
また、`condition_on_text_encodings` という変数で $`\boldsymbol{y}`$ の情報を使うか否かを選べるようになっている。

---

### 演習問題6.5
実装 [https://github.com/openai/glide-text2im/blob/69b5307/glide_text2im/text2im_model.py](https://github.com/openai/glide-text2im/blob/69b5307/glide_text2im/text2im_model.py) を読み解いていけばよい。
ここで定義されている `Text2ImUNet(UNetModel)` クラスの `forward` メソッドは以下の通りである。
UNet の各ブロックにおいて `emb` や `xf_out` が渡されていて、前者はタイムステップの埋め込みに `self.get_text_emb` から得られる `xf_proj` を足したもので、後者は　`self.get_text_emb` から得られる `xf_out` である。

```python
   def forward(self, x, timesteps, tokens=None, mask=None):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.xf_width:
            text_outputs = self.get_text_emb(tokens, mask)
            xf_proj, xf_out = text_outputs["xf_proj"], text_outputs["xf_out"]
            emb = emb + xf_proj.to(emb)
        else:
            xf_out = None
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, xf_out)
            hs.append(h)
        h = self.middle_block(h, emb, xf_out)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, xf_out)
        h = h.type(x.dtype)
        h = self.out(h)
        return h
```

`get_text_emb` メソッドは以下の通りで、本質的には `xf_out` は Transformer エンコーダーの最終層の出力であり、`xf_proj` は最後のトークン出力（を後でタイムステップの埋め込みに加算できるように次元を変換したもの）である。

```python
   def get_text_emb(self, tokens, mask):
        assert tokens is not None

        if self.cache_text_emb and self.cache is not None:
            assert (
                tokens == self.cache["tokens"]
            ).all(), f"Tokens {tokens.cpu().numpy().tolist()} do not match cache {self.cache['tokens'].cpu().numpy().tolist()}"
            return self.cache

        xf_in = self.token_embedding(tokens.long())
        xf_in = xf_in + self.positional_embedding[None]
        if self.xf_padding:
            assert mask is not None
            xf_in = th.where(mask[..., None], xf_in, self.padding_embedding[None])
        xf_out = self.transformer(xf_in.to(self.dtype))
        if self.final_ln is not None:
            xf_out = self.final_ln(xf_out)
        xf_proj = self.transformer_proj(xf_out[:, -1])
        xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL

        outputs = dict(xf_proj=xf_proj, xf_out=xf_out)

        if self.cache_text_emb:
            self.cache = dict(
                tokens=tokens,
                xf_proj=xf_proj.detach(),
                xf_out=xf_out.detach() if xf_out is not None else None,
            )

        return outputs
```

続いて、UNet の各ブロックに渡される `emb` と `xf_out` がどのように使われるかを調べよう。
これは [https://github.com/openai/glide-text2im/blob/69b5307/glide_text2im/unet.py](https://github.com/openai/glide-text2im/blob/69b5307/glide_text2im/unet.py) で定義されている。
`emb` と `xf_out` が使われているのは `TimestepEmbedSequential` クラスにおいてであり、以下のように前者の変数は `TimestepBlock` で、後者の変数は `AttentionBlock` で使われていることが分かる。

```python
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, encoder_out=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, AttentionBlock):
                x = layer(x, encoder_out)
            else:
                x = layer(x)
        return x
```

まずは `emb` が渡される `TimestepBlock` の方から見よう。
これは抽象インターフェースを提供しているクラスで具体的な実装は `ResBlock(TimestepBlock)` クラスにあり、このクラスの `forward` メソッドは以下の通りである。
細かい定義は置いておいて、`self.use_scale_shift_norm` がある場合は scale, shift の形で特徴量に取り込まれ、そうでない場合は単純に特徴量に加算されていることが分かる。
本書では `self.use_scale_shift_norm` がある場合を記述しており、図 6.8 において AdaLN 的に使用と表現した部分である。

```python
    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h
```

次に `xf_out` が渡される `AttentionBlock` の方を見よう。
`AttentionBlock` クラスの `forward` メソッドは以下の通りである。
この中の `self.attention` では `QKVAttention` クラスの `forward` メソッドが適用され、細かい shape の話をスキップすれば、Transformer エンコーダーの情報は交差注意におけるキーとバリューにトークン列方向連結（トークン列方向に連結することでテキストの情報も参照する注意機構を構築している）されている。
本書の図 6.8 において各注意層のキー・バリューに対してトークン列方向に結合と表現した部分である。

```python
    def forward(self, x, encoder_out=None):
        b, c, *spatial = x.shape
        qkv = self.qkv(self.norm(x).view(b, c, -1))
        if encoder_out is not None:
            encoder_out = self.encoder_kv(encoder_out)
            h = self.attention(qkv, encoder_out)
        else:
            h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(b, c, *spatial)
```

`QKVAttention` クラスの `forward` メソッドの実装。

```python
    def forward(self, qkv, encoder_kv=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = th.cat([ek, k], dim=-1)
            v = th.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)
```

これで、GLIDE において Transformer エンコーダーの情報が画像生成を担う UNet にどのように渡されているかの具体的な実装が確認できた。

---

### 演習問題6.6
本書の図 6.13 の生成画像を Google AI Studio [https://aistudio.google.com/](https://aistudio.google.com/) で Nano Banana (models/gemini-2.5-flash-image-preview) モデルを使って再生成してみる。
以下の結果は 20250901 時点で生成した結果である。

`Please draw three different pictures described by the following prompt: a red cube on top of a blue cube` という入力テキストで生成した結果が以下である。
マルチモーダルなモデルなので、画像生成の命令だと分かりやすいようにするのと複数画像を生成させるように `Please draw three different pictures described by the following prompt:` をつけている。
位置関係や色が指示通りの画像が生成されている。

![](/figure/exercise-6-6-1.png)

`Please draw three different pictures described by the following prompt: A sign says deep learning` という入力テキストで生成した結果が以下である。
文字が正しく生成されている。

![](/figure/exercise-6-6-2.png)

`Please draw three different pictures described by the following prompt: A high quality photo of Times Square` という入力テキストで生成した結果が以下である。
画像が 2 枚しか生成されていないが、複雑な画像をある程度うまく生成している。
電光掲示板は図 6.13 の結果よりもかなり現実に近いものが生成されているが、謎の文字列や画像が不自然な点は多く、このレベルになるとまだ完璧な生成は実現されていないことも見て取れる。

![](/figure/exercise-6-6-3.png)

本書で述べた限界についてはかなり改善されているが、まだ複雑な背景の詳細部分の描写は不十分であることが分かる。
これ以外にも様々な観点があるので、興味がある場合には様々な画像生成をしてモデルの振る舞いの理解を深めるとよいだろう。

---

### 演習問題6.7
原論文 [https://arxiv.org/abs/2210.09276v3](https://arxiv.org/abs/2210.09276v3) の 4.5. Limitations と Figure 10 にうまく画像編集ができなかった場合が記述されている。

主な失敗例は 2 つあると述べられており、1 つは望ましい編集がわずかにしか反映されずにターゲット文と十分に整合しない場合、もう 1 つは編集自体はうまく適用されてもズームやカメラアングルといった外在的な画像の詳細に影響が及ぶ場合、である。
これらの具体例が以下で引用した Figure 10 であり、上段が望ましい編集が十分にされていない場合で、下段がズームやカメラアングルに影響が及んでいる場合となっている。

![](/figure/exercise-6-7.png)

これの原因についてはあまり考察がされていない（色々な要素が絡むので単純な要因に帰結できない可能性も高い）が、以下のように言及されている通り、事前学習済みの text-to-image モデルの性能に依存しているのは無視できない影響があると思われる。
例えば、元にしている Imagen モデルの論文 [https://arxiv.org/abs/2205.11487v1](https://arxiv.org/abs/2205.11487v1) の Table 2 で示唆されているように、Imagen では人物の顔に関して画像生成の性能が低くなることが言及されている（言及している割には Imagic の原論文ではそのケースでの実験結果がないのはイマイチだが）。

> since our method relies on a pre-trained text-to-image diffusion model, it inherits the model’s generative limitations and biases. 

---

### 演習問題6.8
Image-Text Contrastive（ITC）損失に関わる部分だけをかいつまんで [https://arxiv.org/abs/2107.07651v2](https://arxiv.org/abs/2107.07651v2) を読み解いていく。

まず、この論文の Figure 1 から引用した提案手法の全体図は以下のもので、これの Image-Text Contrastive Loss の部分が理解したい対象である。

![](/figure/exercise-6-6-8.png)

画像のエンコーダーとテキストのエンコーダーの両方に存在する [CLS] トークンの出力を取り出し、これらを特徴量空間で似た特徴量とするようにエンコーダーを学習することを目的とした損失である。

CLIP と近いが、異なるのは図において Momentum Model と書かれたモデルの存在で、これは本書の 5.3 節でも登場した指数移動平均でパラメーターを更新した画像のエンコーダーとテキストのエンコーダーモデルである。
例えば、画像エンコーダーのパラメーターを $`\theta`$、モメンタム版の画像エンコーダーのパラメーターを $`\theta'`$ としたとき、通常の勾配降下法で $`\theta`$ を更新した後に $`\theta' \rightarrow m \theta' + (1 - m) \theta`$ としてモメンタム版のエンコーダーをアップデートする。
これはテキストエンコーダーにおいても同様である。

画像エンコーダーとテキストエンコーダーの [CLS] トークンの出力をそれぞれ $`\boldsymbol{v}_{\mathrm{cls}}, \boldsymbol{w}_{\mathrm{cls}}`$ とし、モメンタム版の画像エンコーダーとテキストエンコーダーの [CLS] トークンの出力をそれぞれ $`\boldsymbol{v}'_{\mathrm{cls}}, \boldsymbol{w}'_{\mathrm{cls}}`$ とする。
さらに、これらの特徴量を低次元（256次元）の規格化された特徴量に変換する関数をそれぞれ、$`g_v, g_w, g'_v, g'_w`$ とする。

この通常のエンコーダーとモメンタム版のエンコーダーを組み合わせて $`s(I,T) = g_v (\boldsymbol{v}_{\mathrm{cls}})^{\mathsf{T}} g'_w (\boldsymbol{w}'_{\mathrm{cls}})`$ と $`s(T,I) = g_w (\boldsymbol{w}_{\mathrm{cls}})^{\mathsf{T}} g'_v (\boldsymbol{v}'_{\mathrm{cls}})`$ を定義する。
さらに、CLIP では同一バッチの中でのみ画像とテキストの組み合わせを作って損失を計算していたが、M 個の要素を格納する FIFO キュー（これはバッチサイズよりも十分大きくする）を準備し、モメンタム版の出力を複数の更新ステップに渡って保持して損失の計算に用いる。

 ここからは CLIP と同様であり、学習可能なスケール因子 $`\tau`$ を導入して、以下のように softmax で規格化した image-to-text と text-to-image の類似度を定める。

```math
\begin{align}
p_{m}^{\mathrm{i2t}} (I) = \frac{ \exp (s(I,T_m) / \tau ) }{ \sum_{n=1}^{M} \exp (s(I,T_n) / \tau ) }, \ \ p_{m}^{\mathrm{t2i}} (T) = \frac{ \exp (s(T,I_m) / \tau ) }{ \sum_{n=1}^{M} \exp (s(T,I_n) / \tau ) }
\end{align}
```

正解ラベル $`\boldsymbol{y}^{\mathrm{i2t}} (I), \boldsymbol{y}^{\mathrm{t2i}} (T)`$ を正例ペアであれば 1 で負例ペアであれば 0 のワンホットベクトルとすると、ITC 損失は以下で定義される。

```math
\begin{align}
L_{\mathrm{itc}} = \frac{1}{2} \mathbb{E}_{(I,T) \sim D} \left[ \mathrm{CrossEntropy} (\boldsymbol{y}^{\mathrm{i2t}} (I), \boldsymbol{p}^{\mathrm{i2t}} (I)) + \mathrm{CrossEntropy} (\boldsymbol{y}^{\mathrm{t2i}} (T), \boldsymbol{p}^{\mathrm{t2i}} (T)) \right]
\end{align}
```

CLIP との違いはモメンタム版との組み合わせとキューの使用であり、これにより大規模バッチを使わずとも学習の安定性とノイズに対する頑健性が高められるとされている。
この手法は元々は論文 [https://arxiv.org/abs/1911.05722v3](https://arxiv.org/abs/1911.05722v3) で提案されたものであり、ITC 損失はこれを踏襲したものである。

ITC 損失をさらに改善させるものとして図で示されている momentum distillation も用いているが、この演習問題では割愛する。

---

### 演習問題6.9
公式実装 [https://github.com/salesforce/BLIP](https://github.com/salesforce/BLIP) において画像に対する質問応答をするモデルの実装は [https://github.com/salesforce/BLIP/blob/3a29b74/models/blip_vqa.py](https://github.com/salesforce/BLIP/blob/3a29b74/models/blip_vqa.py) にあるので、これを本書の図 6.18 と対応させながら読み解いていく。

`BLIP_VQA` クラスの初期化メソッドは以下である。
画像特徴量を抽出する Image Encoder は `self.visual_encoder` で具体的なモデルは Vision Transformer であり、画像情報を踏まえつつテキスト特徴量を抽出する Image-grounded Text encoder は `self.text_encoder` で具体的なモデルは BERT であり、画像特徴量とテキスト特徴量を融合させて回答テキストを出力する Image-grounded Text decoder は `self.text_decoder` で具体的なモデルは `BertLMHeadModel` である。
`BertLMHeadModel` は [https://github.com/salesforce/BLIP/blob/3a29b74/models/med.py#L811](https://github.com/salesforce/BLIP/blob/3a29b74/models/med.py#L811) で定義されているが、通常の BERT と異なるのは、注意層の変更（生成のためのマスク処理、交差注意で画像情報を取り込めるようにする処理）と言語モデルとしての変更（次トークン予測のための出力追加や shifted right の処理）などである。

```python
class BLIP_VQA(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 480,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                   
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1)
        self.tokenizer = init_tokenizer()  
        
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False) 
        
        decoder_config = BertConfig.from_json_file(med_config)        
        self.text_decoder = BertLMHeadModel(config=decoder_config)          
```

これらのモデルをどのように組み合わせて予測をしているのかを見よう。
`forward` メソッドは以下であり、次の流れでモデルを組み合わせて画像に対する質問応答のテキストを生成していることが分かる。

- 画像特徴量の抽出を `self.visual_encoder` で実施し、出力 `image_embeds` を得る
- 画像とテキストを融合させた特徴量の抽出を `self.text_encoder` で実施し、これの入力は質問テキストと交差注意に取り込まれる `image_embeds` であり、出力  `question_output` を得る
  - コードを追っていくと `BertSelfAttention` クラス [https://github.com/salesforce/BLIP/blob/3a29b74/models/med.py#L97](https://github.com/salesforce/BLIP/blob/3a29b74/models/med.py#L97) に辿り着いて確かに交差注意を使っていることが分かる
- 生成時はビームサーチのための処理を挟んでいるが、本質的には、[BOS] トークンから始まり、交差注意で `question_output` を取り込んだ `self.text_decoder` で回答テキストを生成している

```python
    def forward(self, image, question, answer=None, n=None, weights=None, train=True, inference='rank', k_test=128):
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        question = self.tokenizer(question, padding='longest', truncation=True, max_length=35, 
                                  return_tensors="pt").to(image.device) 
        question.input_ids[:,0] = self.tokenizer.enc_token_id
        
        if train:
            # ここは省略
        else: 
            question_output = self.text_encoder(question.input_ids, 
                                                attention_mask = question.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                                    
                                                return_dict = True) 
            
            if inference=='generate':
                num_beams = 3
                question_states = question_output.last_hidden_state.repeat_interleave(num_beams,dim=0)
                question_atts = torch.ones(question_states.size()[:-1],dtype=torch.long).to(question_states.device)
                model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask":question_atts}
                
                bos_ids = torch.full((image.size(0),1),fill_value=self.tokenizer.bos_token_id,device=image.device)
                
                outputs = self.text_decoder.generate(input_ids=bos_ids,
                                                     max_length=10,
                                                     min_length=1,
                                                     num_beams=num_beams,
                                                     eos_token_id=self.tokenizer.sep_token_id,
                                                     pad_token_id=self.tokenizer.pad_token_id, 
                                                     **model_kwargs)
                
                answers = []    
                for output in outputs:
                    answer = self.tokenizer.decode(output, skip_special_tokens=True)    
                    answers.append(answer)
                return answers
            
            elif inference=='rank':
                # ここは省略
```

これでどのようにモデルを組み合わせて使っているかが理解できたが、改めて本書の図 6.18 を見ると実装と異なっているように見える。
具体的には、図 6.18 では Image-grounded Text decoder の交差注意に取り込まれているのは Image Encoder の出力だが、実装では Image-grounded Text encoder の出力であった。
合わせて、図 6.18 では Image-grounded Text decoder の入力にテキストを渡しているが、実装では [BOS] トークンしか渡していない。

これは図 6.18 のモデル構造は画像に対する質問応答のためのものではなく、画像のキャプションテキストを生成するためのモデル構造を想定しているためで、複数の用途で使える場合はこのように取り扱い方が異なるので、正確な理解のためには実装を読むのがよい。
ちなみに、画像のキャプションテキストを生成するモデルは [https://github.com/salesforce/BLIP/blob/3a29b74/models/blip.py](https://github.com/salesforce/BLIP/blob/3a29b74/models/blip.py) で定義されており、これはまさに図 6.18 と同一である。

---

### 演習問題6.10
LLaVA の論文 [https://arxiv.org/abs/2304.08485v2](https://arxiv.org/abs/2304.08485v2) では様々な具体的なタスクを解いているが、ここでは 5.1 Multimodal Chatbot の例を見てみよう。
以下の図は論文の Figure 3 からの引用で、マルチモーダルなチャットボットの実例を示したものである。

![](/figure/exercise-6-10.png)

まず、最初の入力（$`t = 1`$）では、画像の入力 $`X_{v}`$ として冷蔵庫の中身の画像と、ユーザーの入力 $`X_{q,1}`$ として `What are the meals that I can cook with these?` というテキストとが、$`X_{\mathrm{instruct},1} = [X_{v};X_{q,1}]`$ の形で与えられている。

これに対するモデル $`p_\theta`$ の出力は、本書の図 6.19 と式 (6.6)に基づいて $`p(X_{a,1} \mid X_{\mathrm{instruct},1}) = \prod_{i=1}^L p_\theta (x_i \mid X_{\mathrm{instruct},1}, X_{a,1,<i})`$（$`L`$ は回答のトークン長で、実際は [EOS] トークンを出力するまでだが便宜的にこう書いている。また、次のトークンを逐次的に生成するために $`X_{a,1,<i}`$ でそれまでに生成したテキストを入力している。）を計算しており、`With the variety of food items ...` を返している。

LLaVA は一回のやり取りではなく複数回のやり取りが可能（マルチターンと呼ばれる）なモデルなので、この回答を受けてさらにユーザーが $`X_{\mathrm{instruct},2} = X_{q,2}`$ として `I would like to do the fruit salad. Show me the recipe.` をモデルに渡し、$`p(X_{a,2} \mid X_{\mathrm{instruct,1}, \mathrm{instruct,2}}) = \prod_{i=1}^L p_\theta (x_i \mid X_{\mathrm{instruct},1}, X_{a,1}, X_{\mathrm{instruct},2}, X_{a,2,<i})`$ で `Certainly! ...` を返している（左辺の条件に $`X_{a,1}`$ が含まれていないが、これはユーザーが外部から与えるものではなくモデルが出力したものであるためこのような表記にしている）。

論文の表記のみだとやや不明瞭なところもあるが、具体例を持ち出して対応づけてみると理解が容易になるだろう。
