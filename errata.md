# 原論文から解き明かす生成AI 正誤表

ページ数は紙書籍のものです。

| ページ | 誤 | 正 | 修正対応 | コメント |
| --- | --- | --- | --- | --- |
| 36 | I love reproducibility | I Love Reproducibility（これに合わせて表 2.3 の love と reproducibility の頭文字の l,r は全て大文字になります） | N/A | 筆者が手元で試しているトークナイザーでは頭文字を大文字にしないとサブワードの例が再現されないためです。 |
| 36 | b'\x49', b'\x20', b'\x6c', ... | b'\x49', b'\x20', b'\x4c', ... | N/A | 1 つ上の修正に合わせた修正になります。 |
| 36 | I, ␣love, ␣reproduc, ibility | I, ␣Love, ␣Reprodu, cibility | N/A | 筆者が手元で試しているトークナイザーの分割はこの結果を返します。 |
| 165 | 例えば https://www.youtube.com/watch?v=cHRdyed4-yc 参照するとよい。 | 例えば https://www.youtube.com/watch?v=cHRdyed4-yc を参照するとよい。 | N/A | 脚注の 42) です。 |
| 170 | $`\mathcal{N}\!\Bigl(x;\,\mu_{\theta,i} \bigl(x_1,1\bigr),\,\sigma_1^{2}\Bigr)\,dx`$ | $`\mathcal{N}\!\Bigl(x;\,\mu_{\theta,i} \bigl(\boldsymbol{x}_1,1\bigr),\,\sigma_1^{2}\Bigr)\,dx`$ | 第2刷 | GitHub の表示の問題でうまく表示できないですが、正しくは $`\boldsymbol{x}_1`$ は太字です。 |
| 179 | $`\theta_{\text{EMA}} = \alpha \times \theta_{\text{EMA}} + (1 - \alpha) \times \theta_{\text{current}}`$ | $`\theta_{\text{EMA}} = \alpha \theta_{\text{EMA}} + (1 - \alpha) \theta_{\text{current}}`$ | N/A | 脚注の 63) です。本書ではスカラーの積には $`\times`$ を使いません。 |
| 217 | 同様に、学習時演算量の場合は以下で表せる。 | この結果に $`C = 6NBS`$ を適用することで、学習時演算量の関係式を以下で表せる。 | N/A | 論理展開が不明瞭な点の修正です。 |
| 217 | ここで、$`C = 6NB`$ である。 | （この文を削除） | N/A | 1 つ上の修正により不要となります。 |
| 239 | $`\mathcal{L}_{\text{aux}}`$ | $`L_{\text{aux}}`$ | N/A | GitHub の表示の問題でうまく表示できないですが、本書では損失に `\mathcal` を使いません。 |
| 241 | $`\mathcal{L}_{k, \mathrm{MTP}}`$ | $`L_{k, \mathrm{MTP}}`$ | N/A | GitHub の表示の問題でうまく表示できないですが、本書では損失に `\mathcal` を使いません。 |
| 242 | $`\mathcal{L}_{\mathrm{MTP}} = \frac{\lambda}{D} \sum_{k=1}^{D} \mathcal{L}_{k, \mathrm{MTP}}`$ | $`L_{\mathrm{MTP}} = \frac{\lambda}{D} \sum_{k=1}^{D} L_{k, \mathrm{MTP}}`$ | N/A | GitHub の表示の問題でうまく表示できないですが、本書では損失に `\mathcal` を使いません。 |
