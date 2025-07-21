# 第2章 演習問題の解答

### 演習問題2.1
ワンホット表現における各要素を等しく取り扱う場合には距離が等しくなるので、等しくない取り扱いをすればよい。
例えば、以下のように要素の順番において異なる重み $`w_k > 0`$ を用いて距離を計算すれば、一般に 2 つのワンホット表現（次元 $`n`$）$`d_i, d_j`$ の距離は異なるものとなる。

$$
\mathrm{dist} (d_i, d_j) = \sqrt{\sum_{k = 1}^n w_k (d_{i, k} - d_{j, k})^2 }
$$

これは定義より明らかに距離の公理（非退化、対称、三角不等式）も満たす。
また、これはマハラノビス距離の対角成分のみを有するものに等しい。
どの位置の要素であるかを区別しているのでそのような状況のみで使い得る距離であるが、このようにワンホット表現でも 2 つのトークン間の距離が異なるものを構築することは可能である。

---

### 演習問題2.2
例として `A「遅くなってごめん」B「やっとね」` という文章を考えよう。
B の「やっとね」は同じ文脈に対して使われているが、待ち侘びていてついに会えたという好意的な意味と解釈することもできるし、長らく待たされていて相手を非難する意味と解釈することもできる。

分布仮説は「単語の意味はその単語が登場する文脈によって形成される」というものであったが、文脈をどこまでと捉えるかについては微妙な点を含んでいる。
この文章の前にも文脈があって、AとBが恋人同士で久しぶりに会うことなどが記述されていれば好意的な意味と解釈できるし、代わりに長く待たされていることへの苛立ちなどが記述されていれば非難の意味と解釈するのが自然になる。
一方で、明確な記述がなくて読み手が好きに解釈できるようになっているかもしれない。
また、大量のデータで学習したテキスト生成モデルであれば、この例の文章だけでも好意的な意味も非難の意味もあり得ることを知っており、学習したデータ分布に基づいてどちらがより尤もらしいかを出力することも可能である。

分布仮説は唯一の数学的定義があるものではなく、具体的に何を指すかは場合によるため、分布仮説の議論をする際にはまずは認識を揃えるとよいだろう。

---

### 演習問題2.3
$`\mathcal{S}_{\texttt{gem}} = \{\texttt{priceless}, \texttt{priceless}, \texttt{jewelry}, \texttt{jewelry}, \texttt{ray}\}, \mathcal{S}_{\texttt{jewel}} = \{\texttt{priceless}, \texttt{priceless}, \texttt{priceless}, \texttt{jewelry}, \texttt{ray}, \texttt{faceting}\}`$ である。

重複排除をする場合（type 条件）は、$` \mathcal{count} (\texttt{gem}_y) = 3, \mathcal{count} (\texttt{jewel}_y) = 4 `$ で $` \texttt{gem}_y \cap \texttt{jewel}_y = \{\texttt{priceless}, \texttt{jewelry}, \texttt{ray} \} `$ であるので、$`M_y = \frac{3}{3} = 1`$ である。

重複排除をしない場合（token 条件）は、$` \mathcal{count} (\texttt{gem}_k) = 5, \mathcal{count} (\texttt{jewel}_k) = 6 `$ で $` \texttt{gem}_y \cap \texttt{jewel}_y = \{\texttt{priceless}, \texttt{priceless}, \texttt{jewelry}, \texttt{ray} \} `$ であるので、$`M_k = \frac{4}{5} = 0.8`$ である。

---

### 演習問題2.4
原論文を読むと、文脈の定義として以下の 4 つを用いている。

- 文内のすべての単語（本書で用いたもの）。
- Lorge Magazine Count (これは The Teacher's Word Book of 30,000 Words. という単語頻度カウントの文献) に基づいて、特定の頻度範囲に入る文内のすべての内容語（名詞・動詞・形容詞・副詞などの実質的な意味を持つ語）。
- 各文の文法構造を考慮して、テーマ単語に最も近接するすべての内容語。
- テーマ単語と最も密接に関連していると判断されたすべての単語。この場合、単語AとCは、単語Cの出現が強く単語Aの出現を暗示し、その逆も同様であると判断された場合、密接に関連していると見なす（これは人間が判断する）。

これらの定義を用いて、単語ペア間の同義度合いを文脈的な重複からどの程度推測できるか、を定量的に調べている。

- まず、重複測定値（$`M_y, M_k`$）は、ある単語ペアが3.0未満の同義度合いを持つ（中程度または低い同義度）、という帰無仮説を検証するための統計量として使う。この 3.0 は本書における図 2.3 の観測から、これ未満とこれ以上で重複測定値の傾きが顕著に変わるので、ここが一つの基準だろうと考えて設定している。
- 片側検定で、第一種の過誤（帰無仮説が正しいにも関わらずそれを棄却してしまうもの）確率を 1% or 5% で固定して、特定の重複測定値を超える低同義度ペアの割合を考えることで、検定棄却点を求める。
- そして、この検定棄却点を超える既知の高同義度ペアの割合を計算し、この割合を仮説が誤りであるときにそれを棄却する確率の推定値とする。これが検出力である。

より具体的には以下のことをしている。
手順にするとやや複雑であるが、文脈の定義を変えながら、同義度が低いが重複測定値が高くなるペアを調べて、その基準よりも高い重複推定値を持つ同義度が高いペアがどれくらい存在するかで文脈の定義の良し悪しを測ろうというものである。

- 同義度合いの評価により、20組のテーマペアが3.0を超えている（これは人間が判定した同義度合いの評価による結果）。
- 第一種の過誤を推定するためにペアの各単語の文セットが異なるグループの被験者によって書かれたすべてのテーマ単語ペア（576組）を考慮して、上記の 20 組を除いた 556 組の低同義度ペアで 1％ または 5％の割合に対して重複測定値の検定棄却点を求めた。
  - 実験では 65 組のペアだけでしかデータを作っていないのでそれらにしか同義度合いのデータは存在しないが、それ以外のペアは明確に同義度合いが低いことを仮定（これはデータを眺めて妥当であると判断しているものと思われる）
  - 65 組のペア以外には同義度合いのデータはないが、コーパス生成の方法から明らかなように重複測定値は計算することができるのでそれは計算して使う
  - 上記を踏まえ、低同義度ペア 45 組の重複測定値と同義度合いのデータがない 511 組の重複測定値も合わせて重複測定値の降順で並べて、あとは上位 1% or 5% が含まれる重複測定値を見て検定棄却点を求めればよい
- 20組の高同義度ペアのうち検定棄却点を超えるペアの割合を求めた（これが検出力）

結果は以下の図の通りである（図は [https://dl.acm.org/doi/10.1145/365628.365657](https://dl.acm.org/doi/10.1145/365628.365657) より引用）。
一番上の Unrestricted（これは文脈とは文であるというもの）で、CW & FW の場合を見てみよう。
原論文には FW が Functional Word (機能語: 接続詞・冠詞・助動詞・前置詞・代名詞など) である説明があるので、CW は説明がないが Content Word (内容語: 名詞・動詞・形容詞・副詞) である。
unleveled は前処理なしで、leveled は原形にするなどの前処理をしているものである。
見方として、$`M_y`$ の 1% のところは 75% の割合で正しく検出（つまり、20組の高同義性ペアのうち15組が重複測定値に基づいて正しく高同義度であると検出できていて、556組の低同義性ペアのうち 5.56組 つまり 6 組は重複測定値に基づいて誤って高同義度であると判断される）している、と解釈するものである。
括弧内の数字は棄却点における重複測定値の値を書いていて、これは原論文には記載されていないが、0 ~ 1 の値を取るものを百分率として扱うように 100 倍しているものと考えられる。
具体的な数字としては 1% の場合は 34 で、本書の図 2.3 における 0.34 の線に相当する（ただし、図 2.3 は同義度合いのデータがない 511 組はプロットされないので完全に対応はしていないので注意）。

![](/figure/2-4-1.png)

データ数はそこまで多くはないが、全体的な傾向として検出力は一定水準以上あり、定量的にも重複測定値が高同義性の単語を検出する力があることが見て取れる。
文脈としては、より人間の知識を活用した下の 2 つの場合の方が検出力が高いことも見て取れる。
詳細は置いておくとして、限られたデータ数においては人間の言語知識を活かすことでより性能を高めることができるというのは妥当であろう。

どの文脈の定義も一定の検出力を示しており、データが大量にあれば人間の言語知識を外から与えなくてもモデルが適切に理解するであろうと考え、本書では文脈として「文内のすべての単語」のみを紹介した。

---

### 演習問題2.5
例として、`ここではきものをぬいでください` は「ここで/履物を/脱いで/ください」とも「ここでは/着物を/脱いで/ください」とも分割できる。
英語のように単語間がスペースのような記号で分かち書きされる場合にはこのような曖昧性は発生しないが、日本語のように分かち書きがされない場合には一般にどこで単語を区切るかで意味が異なり得る。

---

### 演習問題2.6
例として、`龍宮の乙姫の元結の切り外し` という言葉は、サブワード分割として例えば「龍宮/の/乙姫/の/元結/の/切り外し」のように分割できるように見える。
しかし、これはアマモという植物の別名であり、このまとまりで 1 つの意味を成すため、サブワード分割をして小さな要素に分解して考えるには不適切で、サブワード分割は効果的には機能しない。
参考: [https://ja.wikipedia.org/wiki/%E3%82%A2%E3%83%9E%E3%83%A2](https://ja.wikipedia.org/wiki/%E3%82%A2%E3%83%9E%E3%83%A2)

---

### 演習問題2.7
コードを実行して `vocab` を調べると以下が得られる。

```python
{'low</w>': 5, 'low e r </w>': 2, 'newest</w>': 6, 'wi d est</w>': 3}
```

これは出現頻度の大きいトークンのペアを結合していくと、最終的（ここでは 10 回のマージ処理）に元の `vocab` がどのような `vocab` に変わるかを示している。
コードでは出現頻度が大きい順にトークンのペアを表示してどのようにトークンがマージされるかを理解させようとしつつ、この `vocab` によって同時にある文字列が与えられたときにそれがどのようにトークナイズされるかも理解してもらおうとしていると思われる。

ただし、BPE の実装は典型的には出現頻度の大きいトークンペアを辞書として保持してそれを使って入力文をトークナイズすることが多いので、コードにおける `vocab` の意味は少し分かりづらく、本書で書いたように理解促進のために効果的なものにはなっていない。

---

### 演習問題2.8
`learn_bpe.py` のコード全体はこちら https://github.com/rsennrich/subword-nmt/blob/92d6139d07d30e12735a0af9e7f7f925ebe62c54/subword_nmt/learn_bpe.py

ポイントになるのは `prune_stats` https://github.com/rsennrich/subword-nmt/blob/92d6139d07d30e12735a0af9e7f7f925ebe62c54/subword_nmt/learn_bpe.py#L271-L284 で以下のコードであり、これは高速化のために `stats` という出現頻度のペアを保持する辞書から `threshold` 未満のペアを削除するものになっている（ `big_stats` には全ての出現頻度の情報が保持されている ）。

```python
def prune_stats(stats, big_stats, threshold):
    """Prune statistics dict for efficiency of max()

    The frequency of a symbol pair never increases, so pruning is generally safe
    (until we the most frequent pair is less frequent than a pair we previously pruned)
    big_stats keeps full statistics for when we need to access pruned items
    """
    for item,freq in list(stats.items()):
        if freq < threshold:
            del stats[item]
            if freq < 0:
                big_stats[item] += freq
            else:
                big_stats[item] = freq
```

あとは `learn_bpe` https://github.com/rsennrich/subword-nmt/blob/92d6139d07d30e12735a0af9e7f7f925ebe62c54/subword_nmt/learn_bpe.py#L298-L360 の処理を追う。
以下の部分において、初期の `threshold` が最大出現頻度の 1/10 であり、`i = 0` のループ処理の最後における `prune_stats` によって多くのペアが枝刈りされるので、早い段階で `stats` に含まれる最大出現頻度のペアは `threshold` を下回ることになる（最大出現頻度のペアはファイルに書き出されて `stats` から取り除かれていくことに注意）。
そのため枝刈りしたペアを再度考慮して `threshold` も更新して処理を続けるという流れになっている。

```python
    # threshold is inspired by Zipfian assumption, but should only affect speed
    threshold = max(stats.values()) / 10
    for i in tqdm(range(num_symbols)):
        if stats:
            most_frequent = max(stats, key=lambda x: (stats[x], x))

        # we probably missed the best pair because of pruning; go back to full statistics
        if not stats or (i and stats[most_frequent] < threshold):
            prune_stats(stats, big_stats, threshold)
            stats = copy.deepcopy(big_stats)
            most_frequent = max(stats, key=lambda x: (stats[x], x))
            # threshold is inspired by Zipfian assumption, but should only affect speed
            threshold = stats[most_frequent] * i/(i+10000.0)
            prune_stats(stats, big_stats, threshold)
        ...（途中省略）...
        if is_bytes:
            outfile.write(most_frequent[0] + b' ' + most_frequent[1] + b'\n')
        else:
            outfile.write('{0} {1}\n'.format(*most_frequent))
        changes = replace_pair(most_frequent, sorted_vocab, indices, is_bytes)
        update_pair_statistics(most_frequent, changes, stats, indices)
        stats[most_frequent] = 0
        if not i % 100:
            prune_stats(stats, big_stats, threshold)
```

簡単のため `i` が小さいところだけを追ってみたが、全体的な処理はより複雑な条件の絡み合いが生じるので、興味があればより深くコードを読んでみるといいだろう。
様々なヒューリスティックな値が使われているのも眺めているとおもしろいところである。

---

### 演習問題2.9
colab で実装した簡単な例: https://colab.research.google.com/drive/1wom259xtR1ZPnD-ACYqegL98J5euagGm?usp=sharing

これは本書のリスト 2.5 を使ったものだが、元の実装の全体は https://github.com/rsennrich/subword-nmt/blob/92d6139d07d30e12735a0af9e7f7f925ebe62c54/subword_nmt/apply_bpe.py#L276-L342 である。

---

### 演習問題2.10
人為的な例として、各トークンの出現確率が `{情報: 0.3, 通信: 0.2, 情報通: 0.4, 信: 0.1}` である場合に、 `情報通信` をサブワード分割する場合を考える。

最長一致の貪欲法であれば、`情報通` が選ばれ、次は残りの `信` が選ばれ、この場合の確率の積を評価してみると $`0.4 \times 0.1 = 0.04`$ である。
一方でビーム幅が 2 であるビームサーチを用いれば、確率の積がより高くなる `情報` と `通信` が選ばれ、この場合の確率の積は $`0.3 \times 0.2 = 0.06`$ である。

（そうなるように人為的に作った例ではあるが）この場合、`情報/通信` と区切る方が意味として分かりやすく、確率の積としても高い値を持つので、ビームサーチの方が良いサブワード分割を導くと考えられる。

---

### 演習問題2.11
BPE において語彙を構築するために使うテキストの長さを $`N`$ として、構築する語彙サイズを $`M`$ とする。

まずはナイーブな実装を考えよう。
ナイーブな実装とは、テキスト全体を走査して隣接ペアの頻度をカウントし、カウントした情報全体を走査して最頻出ペアを同定し、改めてテキスト全体を走査して最頻出ペアのマージ（置換）を実施し、これを構築する語彙サイズの数だけ愚直に繰り返すというものである。
一回のマージにおける主要な計算量は $`\mathcal{O} (N)`$ であり（最頻出ペアの導出はユニークなペアの数に対して線形時間になるが、これは $`\mathcal{O} (N)`$ 以下）、それを $`M`$ 回繰り返すので、計算量は $`\mathcal{O} (MN)`$ となる。
語彙サイズはハイパーパラメーターであるが、最悪の場合 $`\mathcal{O} (N)`$ であるので、最悪の場合の計算量は $`\mathcal{O} (N^2)`$ となる。
トークナイザーの学習データとして、例えば 100 [GB] 程度使う場合、この計算量では学習が困難である。
そのため、ナイーブな実装を用いる場合、英語のようにテキストを分かち書きができる言語を対象にして、単語単位で BPE を適用する必要がある。

一方で、SentencePiece では実装が効率化されているため、計算量は $`\mathcal{O} (N \log N)`$ となる。
様々な場面で最適化がなされているので正確な計算量はより複雑だが、計算量の比較の観点で重要な点のみに絞り、BPE の学習コード https://github.com/google/sentencepiece/blob/d8f7418/src/bpe_model_trainer.cc を確認してみよう。

ポイントは頻度情報の更新で、ナイーブな実装では毎回カウントし直していたが、これは全てカウントする必要はなく関連する局所的な部分だけを更新すればいい。
メインループの処理 https://github.com/google/sentencepiece/blob/d8f7418/src/bpe_model_trainer.cc#L226C3-L309C25 を読んでみよう。
最初に隣接ペアの情報を取得する部分 https://github.com/google/sentencepiece/blob/d8f7418/src/bpe_model_trainer.cc#L210-L215 は $`\mathcal{O} (N)`$ で実施され、`sid` で特徴付けられる文単位で（入力データは文単位に区切られている）、各文でトークンの出現位置を `symbols_` に記録する。
この位置情報を用いて隣接ペアの頻度をカウントしたりマージを実施したり頻度情報を局所的に更新していくことになり、具体的には以下のような処理をしている。

- 100 イテレーション毎に `UpdateActiveSymbols()` を呼び出し、出現頻度の高いペアに絞って扱うようにする。
  - 実装は https://github.com/google/sentencepiece/blob/d8f7418/src/bpe_model_trainer.cc#L137-L164
  - 頻度のカウントと、部分ソートによる高頻度出現ペアの選出が主たる処理で、典型的には後者の計算量が支配的になる（キャッシュなども絡んでいるので正確には複雑なところもあるが）。部分ソートでは、`symbols_` のサイズ（ $`\mathcal{O} (N)`$ まで大きくなり得る）と上位 $`k`$ 件を抽出するので $`\mathcal{O} (N) \log k`$ という計算量となる。
  - $`k`$ も最悪な場合 $`\mathcal{O} (N)`$ まで大きくなり得るので、最悪な場合の計算量は $`\mathcal{O} (N) \log N`$ となる（特に初回の計算は重い）。
- 最頻出ペア `best_symbol` を同定する。
  - 出現頻度の高いペアの中で頻度をカウントして最大のものを探す。多くの場合で頻度計算の結果を保持している（マージ処理した後は頻度再計算が必要だが、上述の通り 100 回ごとに `UpdateActiveSymbols()` が呼ばれるので、ここで頻度計算が発生する場合は高々 100 回程度）ので、計算量は $`\mathcal{O}(k)`$ で、これは最悪 $`\mathcal{O}(N)`$ である。
- 最頻出ペアで置換して、置換後に頻度カウントを局所的に更新する。
  - 置換は `best_symbol` の出現位置の数だけ発生するので無視できるレベルで、最悪 $`\mathcal{O}(N)`$ となる。
  - 頻度カウントの更新が重要な点で、トークンの出現位置を保持しているので、古いトークンの情報を破棄してマージした新しいトークンの情報に置き換えている。これも同様に無視できるレベルで、最悪な場合は $`\mathcal{O}(N)`$ となる。
  - 最悪な場合は $`\mathcal{O}(N)`$ であるが、これは例えば `ababab....` と続くようなパターンであり、この場合はマージ処理の度にテキスト長が指数的に減少するので、この部分の処理は問題にはならない。

以上により、計算量は最悪の場合 $`\mathcal{O} (N \log N)`$ になる。

---

### 演習問題2.12
演習問題 2.11 の解答における頻度カウントの局所的な更新部分を詳しく見ればよい。
具体的には、以下のように `best_symbol` の登場位置ごとに、その左と右の位置を取得し、古い頻度カウント情報をリセットし、マージ処理をして新しいペアを作っている。
最頻出ペアが `AB` で `aABb` という並びの場合、`aA` と `Bb` の頻度情報を破棄して、新しく `aAB` と `ABb` をペアとして扱っていくことになる。

```c++
    for (const uint64 &encoded_pos : best_symbol->positions) {
      const Position pos = DecodePos(encoded_pos);

      if (symbols_[pos.sid][pos.left] == nullptr) {
        // left index might be NULL (set in the previous iteration)
        // when left_symbol == right_symbol.
        continue;
      }
      CHECK_OR_RETURN(symbols_[pos.sid][pos.right]);

      // We have three bigrams [prev, left], [left, right], [right, next],
      // which are affected with this symbol replacement.
      const int next = GetNextIndex(pos.sid, pos.right);
      const int prev = GetPrevIndex(pos.sid, pos.left);

      // Resets the frequencies of bigrams [prev, left] and [right, next].
      ResetFreq(pos.sid, prev, pos.left, best_symbol);
      ResetFreq(pos.sid, pos.right, next, best_symbol);

      // Merges two symbols.
      symbols_[pos.sid][pos.left] = best_symbol;
      symbols_[pos.sid][pos.right] = nullptr;

      // Makes new symbol bigrams [prev, left] and [left, next].
      AddNewPair(pos.sid, prev, pos.left);
      AddNewPair(pos.sid, pos.left, next);
    }
```

局所的な処理で新しいペアの位置情報が得られており、頻度をカウントする場合はこの位置情報を使ってカウントすればいいので、テキスト全体を走査する必要はない。

---

### 演習問題2.13
本書内でも紹介したように、これは ill-defined な問題である。
典型的には `Hello world.` の可能性が一番高いとは考えられるが、空白の情報が失われているため、各トークンの間にスペースを含めるか含めないかの任意性がある。


---

### 演習問題2.14
典型的には、対象のトークンの前にスペースを含む場合と含まない場合で使われ方や意味が異なることは稀であるので、埋め込み特徴量は類似したものになりやすい。
一方で、それ自体が単語としても使われるし、別の単語の一部にもなるようなトークンの場合、トークンの使われ方や意味は異なるので埋め込み特徴量も異なるものになることもある（例えば `us` など）。

事前学習モデルを用いて実験した結果: https://colab.research.google.com/drive/1XKh0PINgQm2Evb9ACCfqTx36lJXqsH3I?usp=sharing

---

### 演習問題2.15
原論文 https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf においては以下のように記述されている。

> We add an exception for spaces which significantly improves the compression efficiency while adding only minimal fragmentation of words across multiple vocab tokens.

スペースを Punctuation のような記号とは異なり文字として扱うことで、原理的には `hot dog` のようにスペースで区切られるが高頻度で使われる単語のペアなども 1 つのトークンとして扱うことが可能となり、これにより圧縮効率を高めることができる。
例えば SentencePiece では `--split_by_whitespace` というオプションでスペースも含めてトークンを作る場合とスペースで区切ってトークンを作る場合を選ぶことができ、その実験結果も公開している: https://github.com/google/sentencepiece/blob/d8f7418/doc/experiments.md

このような性質は認知されていたが、典型的にはスペースで区切ってトークンを構築し、スペースを跨いで文字をマージしてトークンを作るトークナイザーはそれほど作られてこなかった。
最近では SuperBPE: Space Travel for Language Models https://arxiv.org/abs/2503.13423v2 のように再注目されているため、スペースを跨いでトークンを作るトークナイザーが増えていくかもしれない。

---

### 演習問題2.16
20250721 時点で https://tiktokenizer.vercel.app/ を試すと、例えば以下のような例を見つけることができる。

- ⊥（U+22A5）: これは垂直記号であるが、tiktoken によるトークン化では 2 トークン（159042, 98）に分割される。
- ㄱ（U+3131）: これは韓国語の子音字母であるが、tiktoken によるトークン化では 2 トークン（64658, 109）に分割される。

人間による文字の使い方としては 1 文字で扱われるものであるが、バイト単位でのトークン化によって複数トークンに分割されている。

---

### 演習問題2.17
本書における式変形で特に、$`\log P (\boldsymbol{s})`$ を足し引きして式を整理、の部分が少し混み入った計算をしているので、その部分を明らかにしよう。
計算の出発地点として、以下が与えられている状況である。

```math
L_m(Q \circ \mathrm{enc}(\cdot)) = \mathbb{E}_{P(\boldsymbol{s})} \left[ \sum_{\boldsymbol{t} \in \mathrm{enc} (\boldsymbol{s})} \log \left( \frac{1}{Q_{\mathrm{tok} (\boldsymbol{t})}} \right) \right] \quad\text{where}\quad Q_{\mathrm{tok}}(\boldsymbol{t}_{i}) = \pi \bigl( s_{r(i-1)+1} \bigr) \prod_{\ell=1}^{r-1} P \bigl( s_{r(i-1)+\ell+1} \mid s_{r(i-1)+\ell} \bigr)
```

$`Q_{\mathrm{tok}}(\boldsymbol{t}_{i})`$ を代入して展開していく。
本書の図 2.7 も踏まえつつ、トークンによる表式を文字の表式で読み替えよう。
$`\sum_{\boldsymbol{t} \in \mathrm{enc} (\boldsymbol{s})}`$ は $`1`$ から $`\frac{m}{r}`$ の添え字で表現すれば $`\sum_{i = 1}^{\frac{m}{r}}`$ であるので、以下が得られる。

```math
L_m(Q \circ \mathrm{enc}(\cdot)) = \mathbb{E}_{P(\boldsymbol{s})} \left[ - \sum_{i=1}^{m/r} \log \pi (s_{r(i-1) + 1}) - \sum_{i=1}^{m/r} \sum_{\ell=1}^{r-1} \log P ( s_{r(i-1)+\ell+1} \mid s_{r(i-1)+\ell} ) \right]
```

この期待値の中身を $`A`$ と置くと、$`A + \log P(\boldsymbol{s}) - \log P(\boldsymbol{s})`$ として、$` P(\boldsymbol{s}) = \left( \prod_{i=1}^{m-1} P(s_{i+1} \mid s_{i}) \right) \pi (s_1) `$ なる定常Markov過程を用いると、以下のように整理できる。


```math
\begin{align}
A + \log P(\boldsymbol{s}) &=& - \sum_{i=1}^{m/r} \log \pi (s_{r(i-1) + 1}) + \log \pi (s_1) - \sum_{i=1}^{m/r} \sum_{\ell=1}^{r-1} \log P ( s_{r(i-1)+\ell+1} \mid s_{r(i-1)+\ell} ) + \sum_{i=1}^{m-1} \log P(s_{i+1} \mid s_{i}) \\
&=& - \sum_{i=2}^{m/r} \log \pi (s_{r(i-1) + 1}) + \left( - \log P (s_2 \mid s_1) \cdots - \log P (s_{r} \mid s_{r-1}) + \log P (s_2 \mid s_1) \cdots + \log P (s_{r} \mid s_{r-1}) \right) + \log P (s_{r+1} \mid s_{r}) + \cdots
\end{align}
```

ここで、2 行目の括弧内はマイナスの項が $`\sum_{i=1}^{m/r} \sum_{\ell=1}^{r-1} \log P ( s_{r(i-1)+\ell+1} \mid s_{r(i-1)+\ell} )`$ の $`i=1`$ における $`\ell`$ の和の寄与で、プラスの項が $`\sum_{i=1}^{m-1} \log P(s_{i+1} \mid s_{i})`$ の $`i=1`$ から $`i=r-1`$ までの寄与であり、これらが相殺される。
相殺されない項として、$`\sum_{i=1}^{m-1} \log P(s_{i+1} \mid s_{i})`$ の $`i=r`$ の項が残る。
相殺されるものを全て相殺し、特に最後の $`\log P (s_{m} \mid s_{m-1})`$ が相殺されることに注意すると、以下のようにまとめられる。

```math
\begin{align}
A + \log P(\boldsymbol{s}) &=& - \sum_{i=2}^{m/r} \log \pi (s_{r(i-1) + 1}) + \sum_{i=1}^{m/r - 1} \log P (s_{ri + 1} \mid s_{ri}) \\
&=& - \sum_{i=1}^{m/r - 1} \left( \log \pi (s_{ri + 1}) - \log P (s_{ri + 1} \mid s_{ri}) \right)
\end{align}
```

以上により、$`\log P (\boldsymbol{s})`$ を足し引きをして整理すると、本書で書かれている以下の形にまとめられることが示された。

```math
\begin{align}
L_m(Q \circ \mathrm{enc}(\cdot)) = \mathbb{E}_{P(\boldsymbol{s})} \left[ - \sum_{i=1}^{m/r - 1} \left( \log \pi (s_{ri + 1}) - \log P (s_{ri + 1} \mid s_{ri}) \right) - \log P(\boldsymbol{s}) \right]
\end{align}
```

---

### 演習問題2.18
特に重要な仮定としては以下の 3 つが挙げられるだろう。

- データ生成プロセスが k 次 Markov 過程に従うという仮定
  - これは理論解析の助けとなる仮定だが、例えば小説における伏線回収などの長距離の依存や一般常識などの外部知識への依存などが発生するとき、成り立たなくなる仮定である。
- 定常確率の存在
  - これは理論解析の助けとなる仮定だが、例えば時間とともに固有名詞の意味合いが変化したりスラングの流行り廃りなどが発生するとき、成り立たなくなる仮定である。
- Transformer デコーダー型モデルがトークン単位でユニグラム言語モデルを学習しているという仮定
  - これは本書の図 2.6 でも示されているように実験的な証拠も存在するが、近似的にそう仮定してもいいだろうというもので、理論的に示されていることでもない。
  - 具体例としても、図 2.6 においてトークンの予測確率はトークンの位置で変わる場合があるので、この仮定は完全には正しくない（しかし、この仮定を置いて理論解析をしてみる価値が十分にあるレベルの実験結果ではある）。
