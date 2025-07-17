# 第1章 演習問題の解答

### 演習問題1.1
二値分類問題において、クラスを0（Negative）と1（Positive）とした時、以下の混同行列が構築できる。

|      |   | 予測                | 予測                |
|------|---|---------------------|---------------------|
|      |   | 0                   | 1                   |
| 観測 | 0 | True Negative (TN)  | False Positive (FP) |
| 観測 | 1 | False Negative (FN) | True Positive (TP)  |

このとき、正答率（accuracy）と適合率（precision）と再現率（recall）の定義は以下の通りである。

$$
\begin{align}
\mathrm{accuracy} &=& \frac{TN + TP}{TN + FP + FN + TP} \\
\mathrm{precision} &=& \frac{TP}{FP + TP} \\
\mathrm{recall} &=& \frac{TP}{FN + TP} 
\end{align}
$$

accuracy が高くて precition と recall がともに低いような状況を作りたければ、定義から明らかに、TN が他のものよりも十分に大きくてかつ FP や FN が TP よりも大きければよい。
具体的には、以下のような混同行列であれば accuray = 0.9, precision = 0.2, recall = 0.1 となる。

|      |   | 予測                | 予測                |
|------|---|---------------------|---------------------|
|      |   | 0                   | 1                   |
| 観測 | 0 | 116  | 4 |
| 観測 | 1 | 9 | 1 |

より極端には、例えば $`n`$ を自然数として、TP = 1, FP = FN = $`n`$, TN = $`n^2`$として $`n \rightarrow \infty`$ の極限を取れば、accuray が 1 で precition と recall が 0 に収束する。
accuracy は分布の補正を施さない指標なので、相対的に positive なデータが少ない場合にこのような結果になり得る。
