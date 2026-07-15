# Projection matrix roadmap

この文書にはprojection matrixの未完了項目だけを記録する。実装済みAPIの説明は
[`world.md`](world.md)、再現benchmarkと受け入れ結果は
[`../../benchmarks/README.md`](../../benchmarks/README.md)、保留したQA/PSF圧縮は
[`projection-compression-future.md`](projection-compression-future.md)を参照する。

## 受け入れ済みの前提

- `res`は必須で、`res_mode={"fixed", "auto", "ideal"}`を選ぶ。`auto`では`res`がaxis-wise cap、
  `ideal`で`res=None`と固定`partial_res`を指定した場合だけuncapped計算を許す。
- fully visible voxelの外接球と局所detector/PSF scaleからaxis-wise ideal source resを選択する。
  default `point_source_threshold=1/8`は幾何heuristicであり、任意profileの画像誤差保証ではない。
- partial voxelはvisibilityが不連続なのでadaptive化せず、固定`partial_res`を使う。
- `preflight_projection`でvisibility、res bucket、full sample数、partial sample上限をP構築前に確認できる。
- detector spot/cell面積積分とEye内部位置ごとの局所etendueはproduction pathに実装済みである。
- subvoxel sampleの寄与には親voxel体積をsample数で割った重みが入り、三線形補間と合成される。
- fully-visible側のadaptive cap 5はToyと制限MSTで受け入れ済みとする。数値はbenchmark READMEを参照する。
- `chunk_strategy="optical"`はvisible pointをEye方向空間で並べ替えてもnative `P`を再現する。
  PSF圧縮自体はproductionには含めない。

## 積み残し

| 優先度 | 項目 | 完了条件 | 備考 |
|---|---|---|---|
| P0 | wall visibility / preflight高速化 | d=10で時間とpeak memoryを削減し、projection構築でcacheを再利用 | 次に着手 |
| P1 | partial-cell積分 | 不連続境界に必要な精度と積分方式を決定 | P0と独立に検証可能 |
| P2 | inside境界と未知変数仕様 | outside値の扱いを決め、定数・一次・境界profileを回帰化 | P1のreferenceを使用 |
| P3 | Active voxel compact mapping | 列削減、双方向mapping、projection/transpose/pickleを受け入れ | P2後 |
| P4 | Fine-grid最終受け入れ | d=10--25で時間、memory、画像誤差、inverse problemを評価 | P0--P3後 |
| P5 | Eye内部局所visibility | 大きなslitで必要性を定量判定し、必要な場合だけ実装 | 条件付き |

## P0: wall visibilityとpreflight高速化

制限MST d=10（350×150×70、367.5万voxel）ではpreflightが約448秒かかった。
partial subvoxel積分ではなく、inside頂点生成とwall mesh visibility判定が主な候補である。

1. preflightをinside、wall visibility、res選択、集計に分け、wall-clock時間とpeak memoryを測る。
2. `stl_utils.check_visible`のcandidate triangle生成、巨大boolean配列、`nonzero`、重複する座標変換をprofileする。
3. bounded chunk、空間index、camera/Eye並列、既存cache再利用を個別benchmarkし、効果のあるものだけ採用する。
4. preflight後の`set_projection_matrix`がvisibilityを再計算しないことを大規模caseで回帰確認する。
5. World保存・読込後にvisibility cacheを安全に再利用できるgeometry条件とschema/versionを明記する。

preflightは軽い見積もりではなく、wall付きfine gridでは本計算の一部として扱う。

## P1: partial-cell積分

wall/apertureなしの1-cell Toyを使い、解析的な平面または球面inside条件でcellを切る。
球面はMSTのd=75、25相当の曲率比を含め、距離、境界法線、cell内offsetを振る。
`partial_res=2,3,5,...`を高res referenceと比較し、総光量差とpixel誤差をpartial積分だけに限定して測る。

初期結果では曲率より、不連続境界と規則的sample中心の相対位相がworst-case誤差を支配した。
したがって小さな固定resに普遍的な画像誤差保証を与えない。要求精度を満たさなければ、
境界体積率、adaptive boundary quadrature、または別の不連続積分法を検討する。

## P2: inside境界とprofile仕様

source resを上げて改善できるのは、voxel grid上で表現済みの三線形補間profile `S f`を積分する誤差だけである。
voxel gridが実際のemissionを表現できない誤差とは分離する。

- inside voxelの補間にoutside側の隣接voxel値が混ざる条件を可視化する。
- outside emissionをゼロ固定するか、projection列から除外するか、inverse problemで拘束するか決める。
- 定数profileの総光量保存、内部一次profile、inside境界を横切るprofileを回帰テストにする。
- aperture/wall境界とinside境界を別々に検証する。

## P3: Active voxel compact mapping

torus/inside外、または全cameraから常に不可視なvoxelをprojection matrixの列から除外する。

- compact indexと元voxel indexの双方向mappingを保存する。
- emissionのcompact化、projection、transpose、inverse problem後のgrid復元APIを定義する。
- mapping、visibility、projection列、cacheを含む実byte削減量を測る。
- constant emission、camera別visibility、pickle roundtripを回帰テストにする。

QA/PSF圧縮は、compact mapping後もPの保存量またはmatvecが律速として残った場合だけ再検討する。

## P4: Fine-grid最終受け入れ

少なくともd=25、12.5、10 mmで、preflight各段階の時間、res分布、source sample数、projection構築時間、
peak memory、Pの実byte数、画像誤差、projection/transpose、profile fitting時間を比較する。
d=10のfull bounding gridは約1720万voxelになるため、P0--P3後に一度だけ実行する。

## P5: 条件付き物理拡張

`0.5×4 mm`程度のslitで、Eye中心visibilityとEye内部位置ごとの局所visibilityの画像差を実形状で測る。
要求精度に対して無視できない場合だけEye内部visibilityを実装する。局所etendue計算とは別の課題である。

## 次の実装順

1. wall visibility/preflightをprofileし、d=10の律速とmemory allocationを特定する。
2. visibility cacheを壊さずにbounded処理または空間indexを導入する。
3. partial-cell要求精度と積分方式を決める。
4. inside境界、compact mapping、fine-grid最終benchmarkへ進む。
