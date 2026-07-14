# Projection matrix 改善候補

この文書は、projection matrix の高速化と検証作業で残っている候補を記録する。
現在の実装済み範囲と、今後の仕様検討を混同しないための追跡用メモである。

## 現在地

- detector cellとprojected Eye spotの面積積分、有限Eye内の局所etendue、メモリ上限に基づく
  work chunk処理は実装済みである。
- projection cacheにはschema versionを持たせ、互換性のない旧cacheはvisibilityを残して
  projectionだけを無効化する。subpixelは面積積分中の一時評価点とし、永続化するeye別projection
  と全eye合算projectionはいずれもpixel空間 `(N_pixel, N_voxel)` とする。
- optical bin順に通常のnative sparse `P`を作る非圧縮経路までをproduction候補として残す。
- `I ~= Q R`, `A = R S` によるPSF group化の実装と評価コードは
  [Draft PR #9](https://github.com/ingk-124/multi-pinhole/pull/9) に退避した。0.6.0には含めず、
  native sparse `P`を標準表現とする。
- source `res`自動判定、active voxel compact mapping、Eye内部位置ごとの局所visibilityは未実装である。
- `d=75 mm` MST benchmarkは短時間性能の確認には使用済みだが、全profile・detector res・境界を
  含む数値検証は未完了である。

## 0.6.0で完了した項目

### Projection cache の互換性

- projection計算の数値的意味を表すcache schema versionを `World` に保存する。
- versionが存在しない、または一致しない旧worldでは `_projection` と `_P_matrix` だけを
  無効化し、visibilityなど再利用可能な情報は保持する。
- 古いpickleに存在しない `Screen` の派生値を読み込み時に再構築する。
- 現行形式、versionなし、version不一致について回帰テストを持つ。

## 次に確認する項目

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

#### 2026-07 d=75 mm評価

fully visible voxelについては実装済み。各world軸のvoxel chordを実際にscreenへ投影し、
`max(subpixel pitch, eye_size * (1 + f/Z))`で無次元化する。subpixel pitchだけを分母にすると
detector res=5でほぼ全軸が上限に張り付いたため、有限Eye PSFがdominantという物理を判定へ
含めた。`max_projected_step=0.25`を暫定defaultとするが、これは誤差toleranceではない。

MST `d=75 mm`、detector res=1、2 cameraの比較では、fixed `res=5`に対してadaptive full、
`partial_res=5`の4 profile画像L2誤差は約0.35--0.42%だった。fullの展開sampleは約12%へ減るが、
partialが律速のため計算時間は34.6秒から32.7秒への小幅改善に留まった。一方、メモリ上限内でも
並列worker数に応じて最低限のchunkへ分ける修正により、同じfixed `res=5`を24.9秒へ短縮し、
shape、nnz、行列総和の一致を確認した。

partial voxelをres=5 referenceに対するcolumn相対L1誤差から後付けで最適選択するoracle評価では、
1% toleranceで全2,310 camera-voxel列がres=5を必要とした。3%でもsample削減は約4%、5%で
約22.5%だった。したがってpartial用の複雑な自動判定は実装せず、`partial_res`を明示する。

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

> **Future Work:** 以下の設計・Toy実装・benchmarkはDraft PR #9に保存している。
> tolerance 0.01では多くの条件で保存量削減が小さく、0.1では近距離chunkのfactor化が
> native sparse blockより大きくなるケースが確認された。このため0.6.0の実行経路・公開API・
> cacheにはfactorized operatorを入れない。再開時はDraft PRを最新mainへrebaseして評価する。

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
持つ。三線形補間によって1 voxelが複数groupへ寄与することは、一般の疎行列 `A = R S`
として保持する。

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
versionはfactorized operatorの表現確定後に行う。

wallなし3200 voxel、res 2（25600 samples）、24x24 pixelのwarm serial比較では、既存voxel
chunkがmedian `0.0322 s`、voxel重心binningの未圧縮optical chunkが`0.0334 s`で約3.9%遅かった。nnzはともに
46876、最大要素差は`1.4e-23`未満である。これは圧縮前の2-pass並べ替えコストの小規模基準値で、
MST条件のpeak memory・時間評価を置き換えるものではない。

処理単位はcamera・Eye・光学chunkとする。各chunkで `I_chunk` と、それに対応する `S_chunk` の
rowを作り、正規化PSFをgroup化して `Q_chunk`, `R_chunk` を得る。続いて
`A_chunk = R_chunk @ S_chunk` を計算したら、`I_chunk`, `R_chunk`, `S_chunk` は破棄できる。
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

`eps == 0`はsingleton groupを作る特殊ケースとして扱わない。clustering、PSF正規化、`Q,R,A`構築を
明示的に全てバイパスし、そのscopeを通常のdirect block `P_chunk = I_chunk S_chunk`として保存する。
これにより厳密計算を要求した場合の余分な計算と`Q+A`の保存増加を避ける。

factorized operatorにはforward `Q @ (A @ f)`だけでなく、inverse problemで必要なtranspose
`A.T @ (Q.T @ g)`、shape、dtype、camera/Eyeごとのmapping、tolerance、最大group誤差を持たせる。
従来のSciPy sparse `P`を要求するAPIとの互換方法と、projection cache versionも同時に決める。

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

1. mainの0.6.0から新規branchを切り、source `res` 自動判定とfully/partialの適応化を実装する
2. `d=75 mm` MST数値検証で固定resと自動resの画像誤差、時間、選択res分布を比較する
3. inside境界仕様を確定する
4. active voxel compact mappingと `d=10--25 mm` 比較を行う
5. 同一のcamera/Eye座標サンプルを使い、回転と `Z_e/f` だけを変えてgroup化を再評価する
6. MST wallあり条件でpost-processing group化を行い、fully/partial境界と許容閾値を決定する
7. 保存量と実行時間に明確な利点がある場合だけDraft PR #9を再開する
8. subvoxel PSF基準の光学chunkで `I_chunk ~= Q_chunk R_chunk`,
   `A_chunk = R_chunk S_chunk` を直接構築
9. factorized operator、transpose、cache、従来のsparse `P`との互換層を実装
10. fine `P`と構築時間・peak memory・保存容量・forward/transpose時間を比較
11. 圧縮が不利な近距離chunkを非圧縮blockとして混在させるoperator表現を追加
12. compact mappingとgroup化だけでは不足する場合に adaptive voxel mergingを検討
13. PD array、任意 Eye 形状、Eye内部位置ごとの局所visibilityの必要性を定量化
