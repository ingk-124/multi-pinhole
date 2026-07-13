# Projection matrix 改善候補

この文書は、projection matrix の高速化と検証作業で残っている候補を記録する。
現在の実装済み範囲と、今後の仕様検討を混同しないための追跡用メモである。

## 現在地

- detector cellとprojected Eye spotの面積積分、有限Eye内の局所etendue、メモリ上限に基づく
  work chunk処理は実装済みである。
- column group化はwallなしモデルでのpost-processing prototypeと奥行き依存評価まで実施済みで、
  productionのprojection構築処理にはまだ組み込んでいない。
- source `res`自動判定、active voxel compact mapping、projection cache version、局所visibilityは
  未実装である。
- `d=75 mm` MST benchmarkは短時間性能の確認には使用済みだが、全profile・detector res・境界を
  含む数値検証は未完了である。

## main 統合前に確認する項目

### Projection cache の互換性

- projection 計算アルゴリズムと cache schema のバージョンを `World` に保存する。
- バージョンが存在しない、または一致しない古い world を読み込んだ場合は、
  `_projection` と `_P_matrix` を無効化して再計算させる。
- 行列 shape の一致だけでは cache の有効性を判断しない。
- 古い pickle に存在しない `Screen` の派生値、特に
  `_subpixel_u_axis` と `_subpixel_v_axis` を読み込み時に再構築する。
- 現行形式、旧形式、不一致バージョンについて回帰テストを追加する。

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

上記の空間的な voxel merging とは別に、似た投影応答を持つfine voxelのcolumnを
group化する。正規化した `P[:, j]` が十分近いvoxelをまとめ、groupの代表応答を `Q`、
fine emission `f` をgroup入力へ圧縮する写像を `A` として

```text
g = P f ~= Q A f
```

と計算する。`A` は各fine voxelについてgroup番号と感度weightを1個ずつ持つ疎な
restrictionとして表せる。これはSVDや直交化ではなく、ほぼ比例するcolumnの代表化であり、
denseな基底行列は作らない。定数profileのfluxは構成上保存する。profileを別のbasis `B`
で `f = B c` とパラメータ化する場合も、forward計算は `g ~= Q A B c` となり、`A` の
適用コストはfine `P` を保持するコストより小さい。

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

の2次元空間をdetector pixel相当、またはそれより細かい角度binに分割する。これはscreen上の
投影中心 `q = q0 - f (xi, eta)` で分けることと同値である。同じ `(xi, eta)` bin内では、
必要に応じて `f / Z_e`、`log(Z_e)`、またはzoom rateで奥行き方向を並べる・bin分けする。
最後に各binをメモリ上限に収まる最大要素数で分割し、これを圧縮候補chunkとする。
実装上は全voxelを連続走査して近傍を探すのではなく、各voxelの `(xi, eta, depth_bin)` から
整数bin IDを計算してbucketへscatterする。各target columnは必ず1個の光学chunkだけに所属する。

角度binとdepth binは類似columnを探す範囲を制限するためのもので、近似を受理する閾値そのもの
ではない。binを細かくしすぎると圧縮候補を見逃すが精度は悪化せず、粗くしすぎても最終的な
column距離判定でrejectまたは再分割される。最初はdetector pixel幅を角度bin幅として使い、
depth binなしでzoom rate順に並べ、計算量が大きい場合だけdepth binを追加する。

この方法なら、world座標では離れていても同じview cone上にあり、ほぼ同じPSFを持つ点を
比較できる。一方、投影方向が異なる点を「配列上で隣だった」という理由だけで同じgroup候補に
入れない。異なるcameraまたはEye、fully visibleとpartial voxel、inside/wall境界をまたぐ点も
最初から別bucketにする。partial voxelはvisibilityによる不連続性があるため、初期実装では
group化しないか、十分な検証が済むまでsingletonとして保持する。

#### projection計算への組み込み

処理単位はcamera・Eye・光学chunkとし、各chunkについて必要なfine columnを一時的に完成させ、
そのchunk内で類似度判定と `Q_chunk`, `A_chunk` の構築を行い、fine columnを破棄する。最後に
`Q_chunk`を横方向に連結し、各voxelのgroup番号とweightをまとめて、そのcamera/Eyeの
factorized operatorを作る。複数cameraを連結した `P_allcam` 自体は共同分解しない。
各cameraの `Q_camera A_camera` を個別に適用し、得られた画像を最後に連結する。

現在の `I @ S` では、あるsubvoxelの寄与が三線形補間によって最大8個のfine columnへ入り、
メモリ制御用work chunkの境界をまたぐ。そのため、work chunkの部分行列をその場で圧縮すると
未完成のcolumn同士を比較することになり、最終 `P` の近似誤差を保証できない。本番実装では
次のどちらかが必要である。

1. 光学chunkに属するtarget columnを先に決め、そのcolumnへ寄与するsubvoxel cellをhalo付きで
   集め、`S[:, target_columns]`に制限して完全な `P_chunk` を作ってから圧縮する。境界haloでは
   一部のray計算が重複するが、最初の実装として追跡しやすい。
2. 現在と同様にsubvoxel rayを一度だけ計算し、COO寄与をtarget columnの光学chunkへrouteして、
   そのchunkの全寄与が揃った時点で圧縮する。重複計算は避けられるが、buffer管理と並列処理が
   複雑になるため第2段階とする。

まず方式1で正しさと実効圧縮率を確認し、haloによる重複計算が支配的な場合だけ方式2へ進む。
factorized operatorにはforward `Q @ (A @ f)`だけでなく、inverse problemで必要なtranspose
`A.T @ (Q.T @ g)`、shape、dtype、camera/Eyeごとのmapping、tolerance、最大group誤差を持たせる。
従来のSciPy sparse `P`を要求するAPIとの互換方法と、projection cache versionも同時に決める。

#### 現在までの評価と次の判定

wallなしのfine grid prototypeは実装済みだが、現状はfine `P` を構築した後に圧縮する
post-processing診断である。この段階ではprojection構築時間やpeak memoryは削減されない。

無回転の評価では、正規化column L1閾値0.1に対するcolumn圧縮率はmedian `Z_e/f`が
約5、15、50の順に1.01、1.82、4.05倍、閾値0.2では1.37、2.34、8.00倍だった。
遠方ほどpinhole PSFが支配的になり、group化が効くという予想と一致する。`Z_e/f ~= 50`では
Gaussian profileの相対L2誤差は閾値0.1で約`1.0e-5`、0.2で約`2.7e-5`だった。

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

1. cache version と旧 pickle migration
2. `d=75 mm` MST 数値検証と detector res の決定
3. inside 境界仕様の確定
4. main への統合
5. source `res` 自動判定と fully/partial の適応化
6. active voxel compact mapping と `d=10--25 mm` 比較
7. 同一のcamera/Eye座標サンプルを使い、回転と `Z_e/f` だけを変えてgroup化を再評価
8. MST wallあり条件でpost-processing group化を行い、fully/partial境界と許容閾値を決定
9. target column基準の光学chunkとhalo方式で `Q_chunk`, `A_chunk` を直接構築
10. factorized operator、transpose、cache、従来のsparse `P`との互換層を実装
11. fine `P`と構築時間・peak memory・保存容量・forward/transpose時間を比較
12. halo重複が律速なら、COO寄与を光学chunkへrouteする方式へ変更
13. compact mappingとgroup化だけでは不足する場合に adaptive voxel mergingを検討
14. PD array、任意 Eye 形状、Eye 内部位置ごとの局所visibilityの必要性を定量化
