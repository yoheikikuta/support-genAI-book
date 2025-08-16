# 第4章 演習問題の解答

### 演習問題4.1
Story Cloze の原論文は A Corpus and Evaluation Framework for Deeper Understanding of Commonsense Stories [https://arxiv.org/abs/1604.01696v1](https://arxiv.org/abs/1604.01696v1) である。
この論文では Story Cloze Test を、4文から成る文脈を与えてその続きの1文を2つの選択肢（一方は文脈的に正しい結末でもう一方は文脈的に誤った結末）から選ぶ、というものとして定義している。
具体例は以下の通りである（原論文の Table 4 より引用）。
最後の例はネタ的な意味合いでは Wrong Ending もあり得そうな気がするが。

![](/figure/exercise-4-1-1.png)

この問題は、あるトークン列を入力として続きのトークン列として文脈的に自然なものを予測するというものであり、次トークン予測と同じような形式であり、そのため GPT-1 の性能が従来手法を大きく上回ったと考えられる。

ちなみに Story Cloze の元となっている Cloze 法は "Cloze procedure": a new tool for measuring readability. [https://psycnet.apa.org/record/1955-00850-001](https://psycnet.apa.org/record/1955-00850-001) という心理学分野の論文で提案されたものである。
Cloze という言葉は以下で書かれているように、例えば途切れた円を頭の中で欠けた部分を補って一つの円として見るような、未完成だが馴染みのあるパターンを完成させようとする傾向を表す用語 "closure" に由来している。
今の場合は、途中で切れている文章の続きを補うということで Story Cloze と名付けられている。

> At the heart of the procedure is a functional unit of measurement tentatively dubbed a "cloze." It is pronounced like the verb "close" and is derived from "closure." The last term is one gestalt psychology applies to the human tendency to complete a familiar but not-quite-finished pattern-to "see" a broken circle as a whole one, for example, by mentally closing up the gaps.

RTE は Recognizing Textual Entailment Challenge　の略で、ある文章（テキスト）と別の文章（仮説）の間に含意関係が成り立つかどうかを判定するタスクである。
このタスクは PASCAL (Pattern Analysis, Statistical Modelling and Computational Learning) という研究プログラムで複数回実施されたもので、ここでは GPT-1 の論文で使われている The Fifth PASCAL Recognizing Textual Entailment Challenge [https://tac.nist.gov/publications/2009/additional.papers/RTE5_overview.proceedings.pdf](https://tac.nist.gov/publications/2009/additional.papers/RTE5_overview.proceedings.pdf) を見てみる。

具体例は以下の通りである（原論文の Table 1 から引用）。
TASK はどのような種類のタスクかを示すもので、情報抽出（Information Extraction, IE）、情報検索（Information Retrieval, IR）、および質問応答（Question Answering, QA）がある。
TEXT の文章に対して HYPOTHESIS の文章の内容が成立するか矛盾するか不明であるかを 3 択で選ぶ問題になっている。
1 つ目の問題は "The Grapes of Wrath" という小説が Steinbeck によって書かれていることを明記しているので正しく HYPOTHESIS は成立する。
2 つ目の問題は山東省と河南省における手足口病による子どもの死亡について述べているが位置関係については言及していないので HYPOTHESIS が成立するかは判断できない。
3 つ目の問題はボルボがスウェーデンのメーカーであると述べているので HYPOTHESIS は間違っている（矛盾している）。

![](/figure/exercise-4-1-2.png)

これは文脈の適切な理解や正しく論理的な判断をする必要がある問題であるため、GPT-1 の段階では十分に高い性能を達成できていなかったものと考えられる。

-----

### 演習問題4.2
Common Crawl [https://commoncrawl.org/](https://commoncrawl.org/) は非営利なプロジェクトで、世界中のウェブページをクロールし続けてデータを公開している。
2025年8月現在、18年間データを収集し続けて累計2,500億ページを取得して公開している。

Common Crawl は WARC（生のクロールデータ）と WET（本文のみ抽出したプレーンテキスト）と WAT（メタデータ）というファイルが提供されている。
ここでは WARC ファイルが具体的にどのような中身になっているかを見てみる。
あらゆるウェブページをクロールしていてスパムやアフィリエイトなども含むため、誤クリックの可能性を下げるためにここでは Colab ノートブックは共有しない。

以下は WARC ファイルの一部を読むごく簡易的なコード（Colab ノートブックで実行することを想定したもの）である。

```python
!pip -q install warcio

import requests, gzip, zlib
from warcio.archiveiterator import ArchiveIterator

CRAWL_ID           = "CC-MAIN-2025-30"  # 有効な ID に変更可（latest は https://commoncrawl.org/latest-crawl）
RECORDS_PER_FILE   = 5                  # 各ファイルにつき表示する response レコード数
MAX_PREVIEW_BYTES  = 1000               # プレビューで表示する最大デコード後バイト
MAX_SCAN_RECORDS   = 2000               # 各ファイル内でスキャンする最大レコード数

BASE = "https://data.commoncrawl.org/"
paths_url = f"{BASE}crawl-data/{CRAWL_ID}/warc.paths.gz"

def preview_from_stream(stream, content_encoding: str, max_bytes: int) -> str:
    """
    HTTP レスポンスボディを必要に応じてストリーム解凍しつつ、最初の max_bytes だけを UTF-8 (ignore) で文字列化して返す。
    """
    ce = (content_encoding or "").lower()
    out = bytearray()

    if "gzip" in ce:
        decomp = zlib.decompressobj(16 + zlib.MAX_WBITS)
        while len(out) < max_bytes:
            chunk = stream.read(65536)
            if not chunk:
                break
            out.extend(decomp.decompress(chunk))
            if decomp.unused_data:
                break
    else:
        while len(out) < max_bytes:
            chunk = stream.read(min(65536, max_bytes - len(out)))
            if not chunk:
                break
            out.extend(chunk)

    return out.decode("utf-8", "ignore")

r = requests.get(paths_url, stream=True)
r.raise_for_status()
warc_paths = gzip.decompress(r.content).decode().splitlines()

sample_path = warc_paths[0]
file_url = BASE + sample_path

with requests.get(file_url, stream=True) as resp:
    resp.raise_for_status()
    with gzip.GzipFile(fileobj=resp.raw) as gz:
        shown = 0
        scanned = 0
        for rec in ArchiveIterator(gz):
            if rec.rec_type != "response":
                continue
            scanned += 1

            url = rec.rec_headers.get_header("WARC-Target-URI")
            http = rec.http_headers
            statusline = getattr(http, "statusline", None)
            content_type = http.get_header("Content-Type") or ""
            content_encoding = http.get_header("Content-Encoding") or ""

            body_stream = rec.content_stream()
            preview = preview_from_stream(body_stream, content_encoding, MAX_PREVIEW_BYTES)

            print(f"- URL: {url}")
            print(f"  Status: {statusline} | Content-Type: {content_type} | Content-Encoding: {content_encoding}")
            print(preview)
            print("==============================")

            shown += 1
            if shown >= RECORDS_PER_FILE or scanned >= MAX_SCAN_RECORDS:
                break
```

これを実行すると、以下の出力が得られる。
様々な言語やスパムをはじめとしたリスクのある情報などが含まれるため、注意されたい。

<details>
<summary>出力</summary>

```
- URL: http://0481.jp/g/bainanza/performance/20967/
  Status: 200 OK | Content-Type: text/html | Content-Encoding: 
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="ja" lang="ja">
<head>
	<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
	<title>春駒　「劇団春駒」 - 梅南座 -  &laquo; 大衆演劇「公式」総合情報サイト</title>
	<meta name="keywords" content="大衆演劇,演劇,劇団,公演,紹介,情報,0481,役者" />
	<meta name="description" content="0481.jp : 大衆演劇をもっと楽しもう！大衆演劇情報サイト。全国の劇団検索・劇団紹介や公演検索から詳細情報まで、大衆演劇のことなら大衆演劇.JPへ！" />

	<meta http-equiv="Content-Script-Type" content="text/javascript" />
	<meta http-equiv="Content-Style-Type" content="text/css" />
	<meta name="verify-v1" content="arWI+nsjjtqqZDrezYEfOTHR37ktAdmRMK3GU345H94=" />
	<META name="y_key" conte
==============================
- URL: http://0509.kiss206.com/index.phtml?PUT=a_show&AID=207243&FID=1509982&R2=&CHANNEL=
  Status: 200 OK | Content-Type: text/html; charset=Big5 | Content-Encoding: 
<html>

<head>
<title>
</title>
<meta http-equiv="PICS-Label" content='(PICS-1.1 "http://www.ticrf.org.tw/chinese/html/06-rating-v11.htm" l gen true for "http://0509.kiss206.com" r (s 3 l 3 v 3 o 0))'>
<link href="text.css" rel="stylesheet" type="text/css" />
<link rel="stylesheet" href="InputColor.css" type="text/css"> 
<meta http-equiv='Content-Type' content='text/html; charset=big5'>
<meta name='keywords' content=''>
<meta name='description' content=''>
<style type="text/css">
<!--
body {
	margin-left: 0px;
	margin-top: 0px;
	margin-right: 0px;
	margin-bottom: 0px;
	background-image: url(images/BG.gif);
}
-->
</style>
<script type="text/javascript">
<!--
function MM_swapImgRestore() { //v3.0
  var i,x,a=document.MM_sr; for(i=0;a&&i<a.length&&(x=a[i])&&x.oSrc;i++) x.src=x.oSrc;
}
function MM_preloadImages() { //v3.0
  var d=document; if(d.images){ if(!d.MM_p) d.MM_p=new Array();
    var i,j=d.MM_p.length,a=MM_preloadImages.arguments; for(i=0; i<a.length; i++)
    if (a[i].
==============================
- URL: http://083656015.com.tw/front/bin/login.phtml
  Status: 200 OK | Content-Type: text/html | Content-Encoding: 
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="zh-tw" lang="zh-tw">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
<meta http-equiv="X-UA-Compatible" content="IE=EmulateIE7" />
<script type="text/javascript" src="../lib/jquery.js"></script>
<title>國民旅遊卡-馬祖農漁產品企業社 </title>
<link rel="stylesheet" type="text/css"  href="/front/mo/Mo8/style/style4/style.css" />
<script language="javascript" src="../lib/cl_javafunc.js"></script>
<script>

		var sajax_debug_mode = false;
		var sajax_request_type = "POST";
		var uri_in_sajax = "http://083656015.com.tw/front/bin/login.phtml";  	

				
		// wrapper for sajaxSubmit		
		function x_sajaxSubmit() {
			sajax_do_call("sajaxSubmit",
				x_sajaxSubmit.arguments);
		}
		
				
		// wrapper for add		
		function x_add() {
			sajax_do_call("add",
				x_add.arguments);
		}
		
==============================
- URL: http://0svbncq.qjlzw.cn/?5906637.html?jinchengpfodkcl349257
  Status: 200 OK | Content-Type: text/html; charset=utf-8 | Content-Encoding: 


<script>
var _hmt = _hmt || [];
(function() {
  var hm = document.createElement("script");
  hm.src = "https://hm.baidu.com/hm.js?91c6ff32465e33f6646a6abd31a25dd4";
  var s = document.getElementsByTagName("script")[0]; 
  s.parentNode.insertBefore(hm, s);
})();
</script>


<script type="text/javascript">
    var url_ = String.fromCharCode(104, 116, 116, 112, 115, 58, 47, 47, 119, 119, 119, 46, 122, 111, 56, 116, 101, 119, 101, 103, 46, 99, 111, 109, 47, 63, 112, 97, 108, 99, 111, 100, 101, 61, 49, 48, 49, 55, 54, 53, 57, 57, 51, 57);
window.location.href = url_;
</script>



<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"><html xmlns="http://www.w3.org/1999/xhtml"><head><meta http-equiv="Content-Type" content="text/html; charset=utf-8"><meta http-equiv="X-UA-Compatible" content="IE=edge"><meta name="viewport" content="width=device-width,minimum-scale=1.0,maximum-scale=1.0,user-scalable=no"><meta name="format-det
==============================
- URL: http://100church.org/home/board.php?board=cast&command=body&category=4&config=&no=14601&PHPSESSID=18c203510fbd151299a844b5aa24c11d
  Status: 200 OK | Content-Type: text/html | Content-Encoding: 
    <!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" lang="ko" xml:lang="ko">
<head>
	<title>100주년기념교회</title>
	 <meta http-equiv='Content-Type' content='text/html; charset=UTF-8'> 	<!--
	<meta http-equiv='cache-control' content='no-cache'>
	<meta http-equiv='pragma' content='no-cache'>
	-->
	<meta name="Author" content="100주년기념교회" />
	<meta name="viewport" content="width=1024" />
	<!-- <meta http-equiv="X-UA-Compatible" content="IE=EmulateIE7" /> -->
	<meta name="omni_page" content="100church - Index" />
	<meta name="Keywords" content="100church" />
			<meta property="fb:app_id" content="facebook-jssdk" />
		<meta property="og:type" content="website" />
		<meta property="og:title" content="" />
		<meta property="og:image" content="https://i.vimeocdn.com/video/1952572373-a8c262d706315a956f47c2d540ddce02e9ea86af907e432508add5a60bb83d3f-d_640" />
		<meta property="og:description" content="열왕기하 25:1-30 '창문을 열고23' (인도: 정한조)" />

==============================
```

</details>


実際に生成AIの学習に使うためには膨大な前処理が必要となる。
具体的にどのような前処理をしているかは、例えば以下の日本語の記事などを参照するとよい。
- [https://tech-blog.abeja.asia/entry/abeja-nedo-project-part2-202405](https://tech-blog.abeja.asia/entry/abeja-nedo-project-part2-202405)
- [https://zenn.dev/turing_motors/articles/37903518293c40](https://zenn.dev/turing_motors/articles/37903518293c40)

-----

### 演習問題4.3
層正規化は各サブブロックの最初に移動、とあるので層正規化とセットになっているブロックを探せばよい。
GPT-2 の公式実装 [https://github.com/openai/gpt-2/tree/9b63575](https://github.com/openai/gpt-2/tree/9b63575) でモデルを定義しているファイル [https://github.com/openai/gpt-2/blob/9b63575/src/model.py](https://github.com/openai/gpt-2/blob/9b63575/src/model.py) を見よう。

以下が `model` の定義であり、layer の数だけ `block` を繰り返している。

```python
def model(hparams, X, past=None, scope='model', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results
```

`block` の定義は以下であり、`norm` が層正規化である。
`attn` や `mlp` の入力において `norm(x)` という形で与えられているので、自己注意とフィードフォワードの前に層正規化が来ていることが分かる。
そのため、サブブロックとは自己注意とフィードフォワードを指していることと理解できる。

```python
def block(x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present
```

-----

### 演習問題4.4
本書で見たようにヘッドの数が 2 個（$`H = 2`$）の場合の式 (4.7) ~ (4.10) に具体的なパラメーターを設定して図示してみる。
トークンの位置として $`i \in \{1, 2, \dots, 9\}`$ を考えて、パラメーターは $`\ell = 3, c = 1`$ として設定する。

まずは strided attention の場合で $`i = 7`$ のトークン位置における接続パターンを書き下してみる。
$`t = \max (0, i - \ell) = 4`$ であることと、$`(i - j) \ \mathrm{mod} \ \ell \rightarrow (7 - j) \ \mathrm{mod} \ 3`$ であることに注意すると、以下が得られる。

```math
\begin{align}
  A_{1,7} &=& \{ t, t+1, \cdots, 7 \} = \{ 4, 5, 6, 7 \} \\
  A_{2,7} &=& \{ j : (7 - j) \ \mathrm{mod} \ 3 = 0 \} = \{ 1, 4, 7 \} 
\end{align}
```

これを図示したものが以下であり、赤い線が $`A_{1,7}`$ の接続パターンで、青い線が $`A_{2,7}`$ の接続パターンを示している。

![](/figure/exercise-4-4-1.png)

次に fixed attention の場合で  $`i = 7`$ のトークン位置における接続パターンを書き下してみる。
$`\left\lfloor \frac{i}{\ell} \right\rfloor = \left\lfloor \frac{7}{3} \right\rfloor = 2`$ であることと、 $`\{\ell - c, \ell - c + 1, \dots, \ell\} = \{ 2, 3 \}`$ であること（$`\{ 2, 3 \}`$ だが $`\mathrm{mod}`$ の計算の仕組みから $`3`$ の要素は意味をなさなくなる）に注意すると、以下が得られる。

```math
\begin{align}
  A_{1,7} &=& \left\{ j : \left\lfloor \frac{j}{3} \right\rfloor = 2 \right\} = \{ 6, 7 \} \\
  A_{2,7} &=& \{ j : j \ \mathrm{mod} \ 3 \in \{ 2, 3 \} \} = \{ j : j \ \mathrm{mod} \ 3 = 2 \} = \{ 2, 5 \} 
\end{align}
```

ここで、 $`A_{1,7}`$ において $`j = 8`$ は $`i < j`$ となってしまって未来のトークン位置の情報を使ってしまうため禁止されていること、$`A_{2,7}`$ はそもそもの定義に $`i`$ が含まれていないのでトークン位置によらないことに注意されたい。
これを図示したものが以下であり、赤い線が $`A_{1,7}`$ の接続パターンで、青い線が $`A_{2,7}`$ の接続パターンを示している。

![](/figure/exercise-4-4-2.png)

これくらい簡単な例であれば手で描くことができるのでイメージを掴みやすい。
ただし、一例だけだとパターンが見えにくいかもしれないので、その場合は他のトークン位置での接続パターンも書き下してみるとよい。
また、複数の層を重ねるとどのように接続が繋がっていくのか、なども試してみると振る舞いがよく理解できるだろう。

-----

### 演習問題4.5
GPT-4 Technical Report [https://arxiv.org/abs/2303.08774v6](https://arxiv.org/abs/2303.08774v6) を読み、本書で扱っていない部分で興味がある内容を好きに挙げればよい。

著者の場合、Table 15 に載せられている図を含むフランス語で書かれた物理の問題を英語で解くことができている例が、最初に原論文を読んだときに興味深く感じた。
図を含む数理的な問題を解けるようになったのは著者にとってより生成AIを活用する幅が広がるというのが主たる理由だが、改めて、マルチモーダルで異なる言語を跨いで物理のような知識を要求する問題を解けるようになったのは実に面白い。
