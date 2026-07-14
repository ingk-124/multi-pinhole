# Projection matrix 改善候補

この文書は、projection matrix の高速化と検証作業で残っている候補を記録する。
現在の実装済み範囲と、今後の仕様検討を混同しないための追跡用メモである。

## 現在地

- detector cellとprojected Eye spotの面積積分、有限Eye内の局所etendue、メモリ上限に基づく
  work chunk処理は実装済みである。
- column group化は、最終 `P` のpost-processing prototypeに加え、subvoxel PSF行列 `I` を
  `I ~= Q R` と圧縮して `A = R S` を作るToy評価と、optical scope単位のhybrid projection
  構築処理まで実装済みである。
- source `res`自動判定、active voxel compact mapping、局所visibilityは
  未実装である。
- `d=75 mm` MST benchmarkは短時間性能の確認には使用済みだが、全profile・detector res・境界を
  含む数値検証は未完了である。

## main 統合前に確認する項目

### Projection cache の互換性

- projection cache schema versionは `World` に保存済みである。
- バージョンが存在しない、または一致しない古い world を読み込んだ場合は、
  `_projection` と `_P_matrix` を無効化して再計算させる。
- 行列 shape の一致だけでは cache の有効性を判断しない。
- 古い pickle に存在しない `Screen` の派生値、特に
  `_subpixel_u_axis` と `_subpixel_v_axis` を読み込み時に再構築する。
- 現行形式、versionなし旧形式、不一致バージョンの回帰テストを追加済みである。

### MST 実形状での数値検証

まず `d = 75 mm` 程度の短時間で計算できる格子を使い、次の emission
profile を投影する。

1. inside で一定、outside でゼロ
2. 小半径方向に滑らかな Gaussian
3. plasma 境界付近まで値を持つ profile
4. 非対称な profile

各 profile について、総光量、pixel ごとの差、相対 L2 誤差、画像断面を比較する。
detector の `subpixel_resolution = 1, 2, 5` について、総和だけでなく画像分布も
収束していることを確認する。rectangle Eye は、旧実装の spot 幅の不具合修正に
よって結果が変わるため、個別に確認する。

検証後、実際の MST 解析スクリプトで detector 側
`subpixel_resolution = 1` を標準値にできるか決定する。元の約 110 万 voxel、
`d = 25 mm` 相当の計算は、正しさを確認した後に最終 benchmark として一度だけ
実行する候補とする。

### inside 境界と未知変数の扱い

- inside voxel の補間に outside 側の隣接 voxel 値が混ざる場合を調べる。
- outside の emission を常にゼロとするのか、projection matrix の列から除外するのか、
  inverse problem 側で拘束するのかを仕様化する。
- 定数 profile の総光量保存と、境界を横切る profile の挙動をテストする。

## Source subvoxel 解像度

### `res` の自動判定

現在の評価スクリプトは診断用であり、ライブラリに `res="auto"` はない。
自動判定では、代表的な visible voxel に対して `res=1` と `res=2`、必要なら
`res=2` と `res=4` の投影結果を比較し、相対誤差が閾値以下になった最小値を採用する。

- 可能なら軸別に `(r_x, r_y, r_z)` を決める。
- fully visible voxel と partial voxel を別々に評価する。
- 判定に使用した voxel、誤差、採用 res、上限到達の有無を診断情報として返す。
- wall や inside 境界にかかる partial voxel は不連続性を含むため、fully visible
  voxel 用の幾何学的な閾値だけで安全と判断しない。

### voxel ごとの適応 res

遠方かつ fully visible で、voxel の投影像が pinhole PSF より十分小さい場合は
`res=1` とし、partial voxel や投影変化の大きい voxel だけ高い res を使う。
実装する場合は res ごとに voxel を bucket 化し、同じ kernel をまとめて実行する。

## Fine voxel grid を現実的に扱う方法

### 格子幅の比較

emission の空間分解能と有限体積積分の精度を分離するため、少なくとも次を比較する。

- `d=25 mm`, source res 1 または 2
- `d=12.5 mm`, source res 1
- `d=10 mm`, source res 1

`d=10 mm` の bounding grid は約 `405 x 405 x 105`、約 1720 万 voxel になるため、
projection 計算時間だけでなく、可視性配列、行列の列数、メモリ使用量が律速になる。

### Active voxel の compact mapping

torus 外、inside 外、または全 camera から常に不可視な voxel を projection matrix の
列から除外し、compact index と元の voxel index の対応を保存する。inverse problem と
可視化では、この mapping を通して fine grid に戻せるようにする。`d=10 mm` 対応では
優先度が高い。

### Adaptive voxel merging / reduced basis

fine grid 上の emission `f` を reduced parameter `c` と mapping `B` により

```text
f = B c
g = P_fine B c
```

と表す。fully visible かつ光学的に区別しにくい隣接 voxel のみをまとめ、定数 emission
と体積を保存する。wall、inside 境界、partial visibility、急峻な emission を持ち得る
領域は統合しない。mapping、各 group の体積、推定した統合誤差を保存し、結果が何を
表すか追跡可能にする。

最初から octree を導入せず、active voxel compact 化と `res` 自動判定の効果を確認した
後に必要性を判断する。

### 投影応答による column group 化 (`P ~= Q A`)

#### 目的と表現

現在のprojection計算を

```text
u = S f
g = I u
P = I S
```

と分ける。`f` はvoxel中心のemission、`S` は三線形補間とsubvoxel体積weight、`u` は
weighted subvoxel emission、`I[:, i]` はsubvoxel点 `i` の完全なPSFである。最終 `P` の
voxel columnではなく、似たsubvoxel PSFを

```text
I ~= Q R
P = I S ~= Q R S = Q A
A = R S
g ~= Q A f
```

とgroup化する。subvoxel PSFの総感度を `s_i = sum(I[:, i])`、正規化形状を
`h_i = I[:, i] / s_i` とする。group `g` の代表 `q_g` は `s_i` 加重平均で総和1に正規化し、
`R[g, i] = s_i`、それ以外を0とする。したがって `R` はsubvoxel点ごとに1非ゼロ要素を
持つ。ただし実装では `R` をCSR行列として構築せず、`int32 group_index[i]` と
`sensitivity[i]` の2配列だけで表現する。`group_index == -1` は感度0の点を表す。
`A[g, v] = sum_{i:g(i)=g} sensitivity[i] S[i, v]` を `S` の非零要素に沿って直接加算するため、
`A` 構築は概ね `O(nnz(S))` である。三線形補間によって1 voxelが複数groupへ寄与することは、
一般の疎行列 `A` としてそのまま保持する。

これはSVDや直交QR分解ではなく、ほぼ比例するPSF columnの代表化である。各 `q_g` の総和を
1にするため、signedな `f` を含む任意の入力についてdetectorの符号付き総和は保存される。
近似されるのはpixel間の分布である。profileを別basis `B` で `f = B c` と表す場合も

```text
g ~= Q A B c
```

とそのまま適用できる。

#### 光学座標による chunk 分割

圧縮はchunkごとに独立に行う。ただし、ここでいう「光学chunk」は現在の
`_calc_voxel_image_for_eye` が使うvoxel indexの連続slice（メモリ制御用work chunk）とは
異なる。world配列上で連続する点を順番にサンプルしたり、flattenしたindexを一定個数ずつ
切ったりせず、camera/Eyeから見た投影方向で候補を分ける。

Eye中心を原点とする座標を `(X_e, Y_e, Z_e)` とする。`Z_e` はEyeからsourceまでの軸距離で、
camera座標では概ね `Z_camera - f` に相当する。まず

```text
xi  = X_e / Z_e
eta = Y_e / Z_e
```

の2次元空間をdetector pixel pitch基準の角度binに分割する。これはscreen上の
投影中心 `q = q0 - f (xi, eta)` で分けることと同値である。同じ `(xi, eta)` bin内では、
必要に応じて `f / Z_e`、`log(Z_e)`、またはzoom rateで奥行き方向を並べる・bin分けする。
最後に各binをメモリ上限に収まる最大展開sample数で分割し、これを圧縮候補scopeとする。
production実装では全subvoxel点を先に展開せず、visible voxelの重心から整数bin IDを計算して
bucketへscatterする。1 voxelは必ず1個のscopeだけに所属するが、scope内で展開した各subvoxel
PSFは個別にgroup判定する。したがってbinningはvoxel単位でも有限体積近似はsubvoxel単位で残る。

bin幅はvisible pointsのAABBをpixel数で等分して決めず、screen原点に固定した格子を使う。
`optical_bin_width_pixels`をscreen pixel pitchに掛け、`1`を1 pixel、`0.5`を半pixel、`2`を
2 pixels相当とする。subpixel resolutionとは独立に指定し、初期defaultは`1`とする。
wall/apertureに対して完全に不可視なvoxelはbinning前に除外する。partial voxelは重心位置でbinへ
入れ、work chunk内でsubvoxelへ展開した後にpoint visibilityとinside maskを適用する。不可視な
subvoxelは`I`と`S`から除外してからgroup化する。screenと重ならずPSF感度が0になる可視点は
PSF計算後に除外する。

角度binとdepth binは類似columnを探す範囲を制限するためのもので、近似を受理する閾値そのもの
ではない。binを細かくしすぎると圧縮候補を見逃すが精度は悪化せず、粗くしすぎても最終的な
column距離判定でrejectまたは再分割される。最初はdetector pixel幅を角度bin幅として使い、
depth binなしでzoom rate順に並べ、計算量が大きい場合だけdepth binを追加する。

実装では、(1) optical bin、(2) compression scope、(3) work chunkを区別する。bin間圧縮は
初期実装では行わない。1個のbinがメモリ上限を超える場合だけ`f / Z_e`順に複数scopeへ分割し、
scope間も圧縮しない。work chunkはPSF計算呼び出しを細かくしすぎないため複数の完全なscopeを
メモリ上限まで束ねるが、同じwork chunk内でも異なるscopeのPSFを同じgroupには入れない。
full/partialでsource resが異なり得るため、binとwork chunkのcostはvoxel数ではなく各voxelの
`prod(res)`の和で見積もる。

voxel重心binningでは、1 voxelの投影範囲がbin境界をまたぐと圧縮候補を取りこぼす可能性がある。
これはPSF距離判定を通るため精度誤差ではなく圧縮率低下として現れる。診断値としてvoxelの投影
AABB幅とbin幅の比`rho = max(delta_u/bin_u, delta_v/bin_v)`を評価し、`rho >= 1`が多い場合だけ
subvoxel binningまたはbin幅拡大を再検討する。

この方法なら、world座標では離れていても同じview cone上にあり、ほぼ同じPSFを持つ点を
比較できる。一方、投影方向が異なる点を「配列上で隣だった」という理由だけで同じgroup候補に
入れない。異なるcameraまたはEyeは必ず別bucketにする。visibilityはgroup化前にsubvoxel点ごとに
適用し、不可視点を `I` と `S` から除く。残った1点ごとの `I[:, i]` は完全なPSFなので、
fully visible voxel由来かpartial voxel由来かだけを理由に分ける必要はない。ただし由来flagは
診断用に保持し、wall/inside境界をまたいだgroupで誤差が増えていないかを可視化する。

#### projection計算への組み込み

圧縮導入前に、optical bin順へ並べ替えたwork chunkごとに通常の`P_chunk = I_chunk S_chunk`を
計算し、既存のvoxel-index chunk版と一致することを確認する。Toyではwork chunk上限を変えても
絶対誤差`1e-14`以下で一致済みである。同じ経路を`World`の実験option
`chunk_strategy="optical"`へ移し、full/partial voxel、bin幅`0.5/1/2 pixels`で既存経路との差が
`1e-13`以下であることも確認した。defaultの`chunk_strategy="voxel"`は変更していない。

現在のoptical経路はvisible voxel重心だけを保持し、work chunk確定後にsubvoxel座標と`S`を生成
するため、全subvoxel座標の事前保持は廃止した。work chunk処理はまだserialである。またprojection
cache keyはchunk strategyを含まないため、optical optionは常に再計算する。parallel化とcache
parameter keyの導入は別課題とする。serialized cacheのschema version管理は実装済みである。

wallなし3200 voxel、res 2（25600 samples）、24x24 pixelのwarm serial比較では、既存voxel
chunkがmedian `0.0322 s`、voxel重心binningの未圧縮optical chunkが`0.0334 s`で約3.9%遅かった。nnzはともに
46876、最大要素差は`1.4e-23`未満である。これは圧縮前の2-pass並べ替えコストの小規模基準値で、
MST条件のpeak memory・時間評価を置き換えるものではない。

処理単位はcamera・Eye・光学chunkとする。各chunkで `I_chunk` と、それに対応する `S_chunk` の
rowを作り、正規化PSFをgroup化して `Q_chunk`, `group_index`, `sensitivity` を得る。続いて
`A_chunk` を `S_chunk` の非零要素から直接加算したら、`I_chunk`, `S_chunk` と一時mappingは破棄できる。
最後に `Q_chunk` を横方向、`A_chunk` を縦方向に連結すれば、全体の `Q`, `A` が得られる。

この順序なら各 `I[:, i]` は1点光源の完全なPSFなので、三線形補間が最大8個のvoxel columnへ
寄与してもchunk境界のhaloは不要である。複数groupから同じvoxelへ入る寄与は `A` の同一columnに
別rowとして残る。複数cameraを連結した `P_allcam` 自体は共同分解せず、cameraごとの
`Q_camera A_camera` を個別に適用して画像を連結する。

初期clusteringは、chunk全体の `s_i` 加重平均代表を作り、全memberとの距離が閾値内なら受理、
超えたら平均からの最遠点と、そこからの最遠点をseedに二分する再帰方式とする。分割探索には
高速な相対L2、最終診断には正規化PSFのL1と相対L2を両方保存する。single-pass leader方式は
代表更新後に全memberを再確認すればdriftを防げるが、Toy評価では再帰方式より遅くgroup数も
多かったため第2候補とする。

library APIでは `algorithm="recursive"|"leader"`、`metric="relative_l2"|"l1"` を独立に
選べるようにした。leader方式も代表更新のたびに全memberを再検査するので、代表のdriftによる
閾値違反は受理しない。grouping中はscopeのPSFを全screen rowについてdense化せず、scope内で
実際に非零値を持つglobal pixel rowのunionだけを抽出して一時的にdense化する。除外されるrowは
scope内の全PSFで厳密に0なので、L1/L2距離は変わらない。代表 `Q` は元のglobal pixel indexへ
配置し、local dense row、展開subvoxel row、global voxel columnを混同しない回帰テストを追加した。
局所dense working setが `max_scope_dense_bytes` を超える場合はscopeを二分する。これはscope境界を
またぐ圧縮候補を失う可能性はあるが、近似誤差を増やさない。

`eps == 0`はsingleton groupを作る特殊ケースとして扱わない。clustering、PSF正規化、`Q,R,A`構築を
明示的に全てバイパスし、そのscopeを通常のdirect block `P_chunk = I_chunk S_chunk`として保存する。
これにより厳密計算を要求した場合の余分な計算と`Q+A`の保存増加を避ける。
library helper `factorize_psf_columns`ではこの分岐を関数冒頭に実装し、入力`I`を疎行列へ変換する
前に`None`を返す単体テストを追加した。`World`のhybrid経路でもblock builderがgrouping呼び出し前に
同じ分岐を行う。

factorized operatorにはforward `Q @ (A @ f)`だけでなく、inverse problemで必要なtranspose
`A.T @ (Q.T @ g)`、shape、dtypeを実装した。camera/Eye mappingはWorld側で保持し、toleranceと
最大group誤差は構築時の診断値として扱う。従来のSciPy sparse `P`が必要な場合は明示的に展開する。

`HybridProjectionOperator`を実装し、保存表現を `P_direct + Q A` とした。正式な演算APIは
`project(f)`と`backproject(g)`で、既存コード向けに `operator @ f`、`operator.T @ g`、`dot()`も
同じ演算へ委譲する。`to_sparse()`は互換性が必要な場合だけ近似行列を明示展開する。
複数scope/Eyeはdirect部を加算し、`Q`を横、`A`を縦に連結する。subpixel-to-pixel transformは
`P_direct`と`Q`の左から掛け、`Q A`自体は展開しない。

`World.set_projection_matrix`では `chunk_strategy="optical"`かつ
`projection_representation="hybrid"`を指定するとhybrid経路を使える。公開optionは
`psf_tolerance`、`psf_metric`、`psf_grouping`、`max_group_fraction`である。work chunkでは複数scopeの
subvoxel点をまとめて1回のPSF計算に渡す一方、groupingはscopeごとに独立に行う。

#### 現在までの評価と次の判定

最終 `P` のpost-processing prototypeは予備評価として残すが、本命は `I ~= Q R`, `A = R S`
である。Toy評価では、手組みした `I @ S` と既存 `World.set_projection_matrix` の `P` が
Frobenius相対誤差 `1e-16`以下で一致し、visibilityと三線形補間の再構成が確認できた。

無回転の評価では、正規化column L1閾値0.1に対するcolumn圧縮率はmedian `Z_e/f`が
約5、15、50の順に1.01、1.82、4.05倍、閾値0.2では1.37、2.34、8.00倍だった。
遠方ほどpinhole PSFが支配的になり、group化が効くという予想と一致する。`Z_e/f ~= 50`では
Gaussian profileの相対L2誤差は閾値0.1で約`1.0e-5`、0.2で約`2.7e-5`だった。

新しいToy sweep（144 voxel、res 1/2/4、wallなし、相対L2閾値0.1、再帰方式）では、
`Z_e/f = 5, 15, 50`を比較した。res 4のsubvoxel PSFは9216本で、代表PSF数はそれぞれ
5772、1623、196本、`N_s/N_g`は1.60、5.68、47.0倍だった。`nnz(I)`に対する
`nnz(Q)+nnz(A)`の圧縮率は0.92、2.48、15.0倍で、近距離ではfactor化が逆に大きくなり得る。
同条件のGaussian画像相対L2誤差は約0.00189、0.00296、0.00221だった。したがって本番では
光学chunkごとに、factor化後のstorageが減らない場合は圧縮せず元のblockを保持する選択も必要である。

現在のproduction候補実装では、毎回direct `P`も作って厳密Bytes比較することを避けるため、
`N_group/N_active_sample < max_group_fraction`の場合だけfactorized表現を採用する。暫定defaultは
`max_group_fraction=0.8`で、`0`はgroupingをせず常にdirect、`None`は非空factorizationを常に保持する。
この比率はstorageの代理指標であり、最終的な閾値はToy/MSTでCSRの実Bytes
`data.nbytes + indices.nbytes + indptr.nbytes`、構築時間、forward/backproject時間を比較して決める。
本番経路で採否のためだけに `P=IS` を追加計算しない。

production経路を直接使う最初のbenchmarkを追加した。wallなし、600 voxel、source res 2、
24x24 pixel、`Z/f=50`、bin幅1、recursive groupingでは、相対L2閾値`0.03/0.1/0.2`の順に
`P bytes / hybrid bytes`は`0.34/0.77/1.67`、Gaussian画像相対L2誤差は
`1.2e-4/1.84e-3/7.27e-3`だった。構築速度比`time(P)/time(hybrid)`は
`0.17/0.41/0.68`であり、この条件ではhybrid構築はまだ遅い。leader方式は代表数を減らす場合が
あるが、同条件の構築速度比は`0.008--0.058`で、production defaultには不適切である。

1152 voxel、res 2、24x24 pixel、閾値0.1では、`Z/f=15`のbyte圧縮率は`0.34`、`Z/f=50`でも
`0.99`だった。一方、600 voxel、res 4、`Z/f=50`では`N_group/N_sample=291/38400=0.0076`まで
減ってもbyte圧縮率は`0.54`だった。したがって`N_group/N_sample`だけでは最終`P`に対する
`Q+A`のstorageを判定できず、暫定`max_group_fraction=0.8`は採否基準として緩すぎる。
`A`のvoxel方向の広がりと、すでにsubvoxelを統合した`P`の疎性が効くためである。

Toyでscopeごとの`I @ S` assemblyは概ね`0.002--0.005 s`、groupingは閾値に応じて
`0.04--0.44 s`だった。この範囲では採否評価用にdirect blockも計算する追加費用は小さい。
ただしMST規模でも同じかを確認するまではproductionの採用条件を変更しない。次の判定点は、
scopeごとのexact direct Bytes比較を導入するか、`I/S`のsupportからdirect Bytesを推定するかである。

MST `d=75 mm`（shape 54x54x14、40824 voxel）、source res 2、2 camera、閾値0.1でも実測した。
partial voxel visibilityをscopeごとに呼ぶ初期実装は9分以上完了しなかったため中断した。
partial subvoxelをwork chunk単位で一括visibility判定し、結果をscopeへ戻すよう変更した後は、
detector res 1で通常のoptical sparseが`2.98 s`、hybridが`9.64 s`だった。left cameraの保存量は
`8.80 MB`対`9.27 MB`で、`P bytes / hybrid bytes = 0.95`だった。1251 scope中factorized採用は
168 scopeである。

実運用のdetector res 5では通常`6.51 s`、hybrid`71.68 s`、保存量は`8.80 MB`対`8.92 MB`
（比`0.987`）、factorized採用は40/1251 scopeだった。当時はsubpixel PSFをそのままgroup化したため、
detector res 5で距離計算のrow数が最大25倍になっていた。

これを受け、`I_pixel = T_pixel<-subpixel I_subpixel`を各work chunkで直ちに作り、pixel空間で
疎行列組み立てとPSF group化を行うよう変更した。`P_pixel = T I S`の線形性を保つため、subpixel
面積積分の精度は維持される。subpixelは一時的なquadrature表現となり、eyeごとの`_projection`も
`(N_pixel, N_voxel)`へ破壊的に変更した。`_P_matrix`はeye演算子の単純な和である。旧仕様はGit履歴で
追跡でき、projection cache schemaを2へ上げたため旧pickleのvisibilityは残してprojectionだけを
再計算する。

同じMST `d=75 mm`、source res 2、閾値0.1を再測定した。今回の実行環境では通常sparseが
detector res 1/5で`9.04/10.36 s`、hybridが`15.39/16.51 s`だった。絶対時間は以前の測定より
全経路で長いため単純比較できないが、旧実装でhybridだけがres 5で`71.68 s`まで増えた現象は消え、
res 1/5のhybrid保存量、factorized scope数、group数はそれぞれ`9,270,320 bytes`、`168/1251`、
`1449`で完全に一致した。すなわち圧縮処理はdetector subpixel数から独立した。

detector res 5で通常sparseとhybridを直接比較すると、行列Frobenius相対誤差は`3.19e-3`、最大要素
絶対誤差は`6.87e-6`、voxelごとの総光量の相対L2誤差は`2.73e-16`だった。総光量は丸め誤差範囲で
保存されている。一方、保存量は通常Pの`8,802,860 bytes`に対してhybridが`9,270,320 bytes`で、
この閾値・scope採用基準ではまだ約5.3%不利である。次はexact Bytesによるscope採否判定が必要。

cameraを30度回転した評価では奥行きに対する単調性が崩れた。現在の `Voxel` はworld軸に
整列しており、camera座標で指定したboxをworld座標のAABBに変換すると横方向分解能と
奥行き分解能が同時に変化するためである。回転依存を分離する次の評価では、同一の
camera/Eye座標サンプルを直接投影する。

採用判断では、constant/linear/square/Gaussian profileについてflux、相対L1/L2、pixelごとの
最大誤差を測る。さらにwallなし回転系、MST wallあり、fully/partial境界、複数cameraで、
projection構築時間、peak memory、保存容量、forward/transpose時間をfine `P` と比較する。
圧縮率だけでなくこれらが改善し、指定toleranceに対する誤差が追跡可能な場合に本番化する。

## 有限サイズ Eye の物理モデル

### Eye 内部位置ごとの局所 etendue

screen上のoverlap位置をEye内の通過位置へ逆写像し、位置ごとの距離、角度、etendueを
積分する処理は実装済み。spotが1 pixelより小さい場合もEye形状上のquadratureを使う。
circle、ellipse、rectangleについて、独立な高次開口積分との回帰テストを持つ。

局所etendueは標準計算として常時使用する。旧来のEye中心光線による一様近似は高速化に
ならず、通常APIにON/OFFのboolは追加しない。旧projectionとの互換比較や補正量の診断が
必要な場合は、テスト・評価コード内で中心近似を計算する。将来どうしても公開切替が必要に
なった場合は、速度オプションではなく `etendue_mode="local" | "eye_center"` のように
物理モデルを明示する。

`d=75 mm` MST benchmark（40824 voxel、source res 1、detector res 1、parallel 4）では、
局所etendue導入前後のwarm実行時間はともに約7.3--7.6秒で、意味のある低速化は
観測されなかった。行列の `nnz` は92643で不変、行列総和は11.31715から11.32026へ
約0.0275%変化した。したがってMSTの小さいpinholeでは補正は小さい一方、
`0.5 x 4 mm`程度のslit、近距離、斜入射では中心近似との差を別途評価する価値がある。

未実装なのは、wallやapertureによるvisibilityがEye内部位置によって変わる場合の局所
visibilityである。必要性を`0.5 x 4 mm`程度のslitと実際のwall配置で評価する。

## Detector と aperture の拡張

### Photodiode array

- active area、pitch、pixel 間の非感光領域を明示できる detector 表現を検討する。
- fill factor が高い配列は連続 screen として扱い、面積積分を利用する。
- active area が spacing より十分小さい疎な配列は、複数の 1x1 screen、または
  multi-detector screen として表す。
- `0.75 x 4 mm`、pitch `0.95 mm`、gap 約 `0.2 mm` のような具体例で検証する。

### Eye 形状

spot の面積積分が直接対応する Eye 形状は circle、ellipse、rectangle である。
任意形状の aperture mesh は遮蔽判定には使えるが、その形状を projected spot の
放射分布として積分する機能とは別である。任意 Eye 形状が必要になった場合は、polygon
clip または数値積分を追加する。

## 意図的に保留している最適化

- `I @ S` の完全融合は保留する。現在の sparse 積は一時的な非ゼロ要素増加を抑えており、
  単純な融合は subvoxel 補間の最大 8 列分を展開して、逆にメモリを増やす可能性がある。
- 全 voxel で一律に高い source res を使う前に、自動判定と適応 res を評価する。
- adaptive voxel merging は、compact mapping だけでは不足すると確認してから着手する。

## 実施順序案

1. ~~projection cache schema version と旧pickleのprojection無効化~~（完了）
2. `d=75 mm` MST 数値検証と detector res の決定（pixel空間group化後を再測定）
3. inside 境界仕様の確定
4. main への統合
5. source `res` 自動判定と fully/partial の適応化
6. active voxel compact mapping と `d=10--25 mm` 比較
7. 同一のcamera/Eye座標サンプルを使い、回転と `Z_e/f` だけを変えてgroup化を再評価
8. MST wallあり条件でpost-processing group化を行い、fully/partial境界と許容閾値を決定
9. ~~subvoxel PSF基準の光学scopeで `I_chunk ~= Q_chunk R_chunk`,
   `A_chunk = R_chunk S_chunk` を直接構築~~（`R`は2配列で表現し、`A`へ直接加算済み）
10. ~~factorized operator、transpose、従来のsparse `P`との明示変換を実装~~（hybrid optionとして完了）
11. fine `P`と構築時間・peak memory・CSR実Bytes・forward/backproject時間を比較し、暫定`0.8`を校正
12. ~~圧縮が不利な近距離scopeを非圧縮blockとして混在させるoperator表現を追加~~
13. compact mappingとgroup化だけでは不足する場合に adaptive voxel mergingを検討
14. PD array、任意 Eye 形状、Eye 内部位置ごとの局所visibilityの必要性を定量化
