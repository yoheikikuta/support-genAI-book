# 原論文から解き明かす生成AI 正誤表

ページ数は紙書籍のものです。
数が多いので、理解の妨げになる誤りとそれ以外の誤り（単純な typo や表記揺れなど）とを分けて書いています。

## 理解の妨げになる誤り

| ページ | 誤 | 正 | 修正対応 | コメント |
| --- | --- | --- | --- | --- |
| 76 | 実装上は複雑なことはなく、数式で書けば以下のようになる。 | 数式で書けば以下のようになる（同じ特徴量から作っているので $`Q,K,V \in \mathbb{R}^{n \times d_{\text{model}}}`$ となることに注意）。 | N/A | これは厳密には誤りではないですが、前節と記号が同じで違うものを扱っていて混乱をきたすものです。[@phys_yoshiki](https://x.com/phys_yoshiki) さんありがとうございます。 |
| 83 | $`W_{1} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}, W_{2} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}`$ | $`W_{1} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}, W_{2} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}`$ | N/A | [@phys_yoshiki](https://x.com/phys_yoshiki) さんありがとうございます。 |
| 112 | $`\mathrm{Attend} (\{\boldsymbol{h}\}, S) = \left( a(\boldsymbol{h}_i, S_i) \right)_{i \in \{i, \dots, n\}}`$ | $`\mathrm{Attend} (\{\boldsymbol{h}\}, S) = \left( a(\boldsymbol{h}_i, S_i) \right)_{i \in \{1, \dots, n\}}`$ | N/A | [@phys_yoshiki](https://x.com/phys_yoshiki) さんありがとうございます。 |

## それ以外の誤り

| ページ | 誤 | 正 | 修正対応 | コメント |
| --- | --- | --- | --- | --- |
| 16 | それゆえに論文という体で出されているにも関わらず | それゆえに論文という体で出されているにもかかわらず | N/A | 逆説の意味なので関わらずだと不適切なので修正です。[@phys_yoshiki](https://x.com/phys_yoshiki) さんありがとうございます。 |
| 16 | https://arxiv.org/abs/1803.11175/v2 | https://arxiv.org/abs/1803.11175v2 | N/A | バージョン番号は `/` なしで付与する必要があります。[@phys_yoshiki](https://x.com/phys_yoshiki) さんありがとうございます。 |
| 26 | フーリエ変換 | Fourier変換 | N/A | 図2.1中の表記で、本書では人物名は英語表記しています。[@phys_yoshiki](https://x.com/phys_yoshiki) さんありがとうございます。 |
| 36 | I love reproducibility | I Love Reproducibility（これに合わせて表 2.3 の love と reproducibility の頭文字の l,r は全て大文字になります） | N/A | 筆者が手元で試しているトークナイザーでは頭文字を大文字にしないとサブワードの例が再現されないためです。 |
| 36 | b'\x49', b'\x20', b'\x6c', ... | b'\x49', b'\x20', b'\x4c', ... | N/A | 1 つ上の修正に合わせた修正になります。 |
| 36 | I, ␣love, ␣reproduc, ibility | I, ␣Love, ␣Reprodu, cibility | N/A | 筆者が手元で試しているトークナイザーの分割はこの結果を返します。 |
| 52 | 前に空白のあるなしにかかわらず | 前に空白のあるなしに関わらず | N/A | 関係なくの意味なのでそれに合わせた修正です。 |
| 56 | データがMarkov過程に基づいているにも関わらず | データがMarkov過程に基づいているにもかかわらず | N/A | 逆説の意味なので関わらずだと不適切なので修正です。 |
| 58 | ユニグラム言語モデル$`Q \in \mathcal{Q}_{\mathrm{1-gram}}`$ | ユニグラム言語モデル$`Q \in \mathcal{Q}_{\text{1-gram}}`$ | N/A | 本書では `\mathcal{Q}_{\text{1-gram}}` で統一しています。[@phys_yoshiki](https://x.com/phys_yoshiki) さんありがとうございます。 |
| 68 | 線型結合 | 線形結合 | N/A | 本書では線形で統一しています。他にも表記揺れしているところがありますが線形で読み替えてください。[@phys_yoshiki](https://x.com/phys_yoshiki) さんありがとうございます。 |
| 72 | レーベンシュタイン距離 | Levenshtein 距離 | N/A | 図3.6中の表記で、本書では人物名は英語表記しています。[@phys_yoshiki](https://x.com/phys_yoshiki) さんありがとうございます。 |
| 72 | query は | クエリーは | N/A | 図3.6のキャプションで、本書ではクエリーとカタカナで統一しています。[@phys_yoshiki](https://x.com/phys_yoshiki) さんありがとうございます。 |
| 73 | $`(i = 1, \dots n)`$ | $`(i = 1, \dots, n)`$ | N/A | 式(3.10)の上で、カンマ漏れです。[@phys_yoshiki](https://x.com/phys_yoshiki) さんありがとうございます。 |
| 76 | $`\mathrm{Concatenate} (\mathrm{head}_1, \cdots, \mathrm{head}_H) W^{{\mathrm{out}}}`$ | $`\mathrm{Concatenate} (\mathrm{head}_1, \dots, \mathrm{head}_H) W^{{\mathrm{out}}}`$ | N/A | 本書では `\dots` に統一しています。他にも表記揺れしているところがありますが `\dots` で読み替えてください。[@phys_yoshiki](https://x.com/phys_yoshiki) さんありがとうございます。 |
| 76 | $`(i = 1, \dots H)`$ | $`(i = 1, \dots, H)`$ | N/A | 式(3.14)の下で、カンマ漏れです。[@phys_yoshiki](https://x.com/phys_yoshiki) さんありがとうございます。 |
| 101 | ブルームフィルター | Bloomフィルター | N/A | 本書では人物名は英語表記しています。[@phys_yoshiki](https://x.com/phys_yoshiki) さんありがとうございます。 |
| 111 | i番目のトークンが前層のどのトークンと接続するかのインデックス | $`i`$番目のトークンが前層のどのトークンと接続するかのインデックス | N/A | `i` の表記は数式中なのでイタリック体です。[@phys_yoshiki](https://x.com/phys_yoshiki) さんありがとうございます。 |
| 140 | N/A | これは最大化する対象であることに注意されたい。 | N/A | 式(4.20)の直後に追加です。これは厳密には誤りではないですが、最小化する損失関数と最大化する目的関数で同じ記号を使っていて混乱をきたすので注意書きを追加しました。[@himkt](https://x.com/himkt) さんありがとうございます。 |
| 160 | フーリエ変換 | Fourier変換 | N/A | 本書では人物名は英語表記しています。[@phys_yoshiki](https://x.com/phys_yoshiki) さんありがとうございます。 |
| 165 | 例えば https://www.youtube.com/watch?v=cHRdyed4-yc 参照するとよい。 | 例えば https://www.youtube.com/watch?v=cHRdyed4-yc を参照するとよい。 | N/A | 脚注の 42) です。 |
| 166 | $`\mathcal{N}\!\Bigl(x;\,\mu_{\theta,i} \bigl(x_1,1\bigr),\,\sigma_1^{2}\Bigr)\,dx`$ | $`\mathcal{N}\!\Bigl(x;\,\mu_{\theta,i} \bigl(\boldsymbol{x}_1,1\bigr),\,\sigma_1^{2}\Bigr)\,dx`$ | 第2刷 | GitHub の表示の問題でうまく表示できないですが、正しくは $`\boldsymbol{x}_1`$ は太字です。 |
| 167 | $`t`$ は 1 から T までの | $`t`$ は $`1`$ から $`T`$ までの | 第2刷 | `T` の表記は数式なのでイタリック体です。[@phys_yoshiki](https://x.com/phys_yoshiki) さんありがとうございます。 |
| 179 | $`\theta_{\text{EMA}} = \alpha \times \theta_{\text{EMA}} + (1 - \alpha) \times \theta_{\text{current}}`$ | $`\theta_{\text{EMA}} = \alpha \theta_{\text{EMA}} + (1 - \alpha) \theta_{\text{current}}`$ | N/A | 脚注の 63) です。本書ではスカラーの積には $`\times`$ を使いません。 |
| 180 | https://arxiv.org/abs/2302.09057 | https://arxiv.org/abs/2302.09057v1 | N/A | バージョン番号の付け忘れです。 |
| 200 | リンゴの予測確率がほぼ0%だったにも関わらず | リンゴの予測確率がほぼ0%だったにもかかわらず | N/A | 逆説の意味なので関わらずだと不適切なので修正です。 |
| 219 | 同様に、学習時演算量の場合は以下で表せる。 | この結果に $`C = 6NBS`$ を適用することで、学習時演算量の関係式を以下で表せる。 | N/A | 論理展開が不明瞭な点の修正です。 |
| 219 | ここで、$`C = 6NB`$ である。 | （この文を削除） | N/A | 1 つ上の修正により不要となります。 |
| 239 | $`\mathcal{L}_{\text{aux}}`$ | $`L_{\text{aux}}`$ | N/A | GitHub の表示の問題でうまく表示できないですが、本書では損失に `\mathcal` を使いません。 |
| 240 | https://arxiv.org/abs/2404.19737 | https://arxiv.org/abs/2404.19737v1 | N/A | バージョン番号の付け忘れです。 |
| 241 | $`\mathcal{L}_{k, \mathrm{MTP}}`$ | $`L_{k, \mathrm{MTP}}`$ | N/A | GitHub の表示の問題でうまく表示できないですが、本書では損失に `\mathcal` を使いません。 |
| 242 | $`\mathcal{L}_{\mathrm{MTP}} = \frac{\lambda}{D} \sum_{k=1}^{D} \mathcal{L}_{k, \mathrm{MTP}}`$ | $`L_{\mathrm{MTP}} = \frac{\lambda}{D} \sum_{k=1}^{D} L_{k, \mathrm{MTP}}`$ | N/A | GitHub の表示の問題でうまく表示できないですが、本書では損失に `\mathcal` を使いません。 |
| 276 | アインシュタイン方程式 | Einstein方程式 | N/A | 脚注29)の表記で、本書では人物名は英語表記しています。 |
