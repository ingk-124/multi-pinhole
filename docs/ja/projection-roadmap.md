# Projection matrix roadmap

この文書には、projection matrixの高精度化・高速化について今後実施する項目だけを記録する。
完了した変更と撤回した設計はGit履歴で追跡し、ここには残さない。

## 積み残しサマリー

| 優先度 | 項目 | 完了条件 | 依存関係 |
|---|---|---|---|
| P0 | Source quadratureの単純化 | 外接球方式へ置換し、ToyとMSTで精度・時間を受け入れ | なし。現在の作業対象 |
| P1 | MST実形状とinside境界 | profile・detector res・境界仕様を回帰テストで固定 | adaptive比較はP0に依存 |
| P2 | Fine gridとcompact mapping | `d=10--25 mm`で時間・peak memory・mapping APIを受け入れ | P0、P1の仕様確定後 |
| P3 | Eye内部局所visibility | slit実形状で必要性を定量判定 | 条件付き。効果がなければ実装しない |

外接球判定のdefault thresholdは `1/8`とする。未決定事項はinside/outside未知変数の扱いであり、
境界profileの可視化とinverse problem側の要件から決定する。

## 1. Source quadratureの単純化

### 目的

fixed `res`を高精度計算と検証のreferenceとして維持し、adaptiveは外接球の投影径からfully visible
voxelの理想軸別resを選ぶ高速化optionとする。幾何判定を任意のemission profileに対する誤差保証とは
扱わない。

- adaptive判定はfully visible voxelだけに適用する。
- partial voxelはadaptive判定から除外し、明示された`partial_res`を使用する。
- pixel境界、profile勾配、距離帯ごとの補正など個別のヒューリスティックを追加しない。
- 判定に使用した無次元比、足切り結果、採用resを診断できるようにする。

### 外接球によるres=1判定

voxelの辺長を `(dx, dy, dz)` とし、外接球の直径

```text
L = sqrt(dx^2 + dy^2 + dz^2)
```

をvoxel内部のworst-case特徴長さとする。Eyeからvoxel中心までの軸距離を `Z_e`、中心視線の
off-axis角を `theta`、焦点距離を `f` とすると、局所投影Jacobianの最大倍率から

```text
projected_diameter ~= f * L / (Z_e * cos(theta))
```

と評価する。`1/cos(theta)`で斜視時のworst-caseを含め、voxelのcornerや3軸両端は実投影しない。
外接球がEye面へ近づきすぎる条件など、局所近似を安全に使えないvoxelはres=1判定を不合格とする。

有限Eye PSFとdetector samplingからscreen 2軸の基準長を作る。

```text
reference_u = max(subpixel_pitch_u, eye_size_u * (1 + f / Z_e))
reference_v = max(subpixel_pitch_v, eye_size_v * (1 + f / Z_e))
reference_size = min(reference_u, reference_v)
rho = projected_diameter / reference_size
```

投影方向を特定しないscalar worst-caseなので、小さい方のreference scaleを採用する。`rho <= 1/8`
なら `res=(1, 1, 1)`とする。それ以外は同じ閾値から立方体に近いsubvoxelを作る理想軸別resを
計算し、指定されたfixed resを上限としてclipする。

`adaptive_source_resolution=True, res=None`ではideal resをそのまま使用する。intまたはtupleの`res`を
指定した場合だけ軸別上限としてclipする。局所近似が無効でidealを有限に決められないvoxelについて、
上限なしの計算は明示的にerrorとし、上限resの指定を要求する。

### scalar resからaxis-wise resへの変換

scalar `res=r`は最長辺方向の分割数と解釈し、subvoxelが立方体に近くなるように変換する。

```text
d_max = max(dx, dy, dz)
h = d_max / r
r_x = clip(ceil(dx / h), 1, r)
r_y = clip(ceil(dy / h), 1, r)
r_z = clip(ceil(dz / h), 1, r)
```

例えばvoxel size `(10, 10, 2) mm`、scalar `res=5`なら `(5, 5, 1)`となる。立方voxelでは
従来どおり `(r, r, r)`となる。

- scalar `res`と`partial_res`には縦横比変換を適用する。
- 明示的なtuple `(r_x, r_y, r_z)`はそのまま使用する。
- voxelごとに辺長が異なる場合は、得られたinteger tupleごとにbucket化してprojection kernelを
  まとめて実行する。

### 実装と受け入れ条件

1. 外接球指標とscalar-to-axis変換を副作用のないutilityとして実装する。
2. 立方・非等方voxel、camera回転、off-axis、近距離、anisotropic Eyeのunit testを追加する。
3. 現在の端点投影による判定と、res=1 voxel集合、判定時間、展開sample数をToyで比較する。
4. wall・apertureなし、全voxel fully visibleのToyで `d` と `Z/f`を掃引し、fixed res referenceに
   対するprojection/profile誤差を比較する。
5. `d=75 mm` MST短時間モデルで判定時間、projection構築時間、res分布、画像誤差を比較する。
6. 外接球方式が十分保守的であることを確認した後、端点投影方式を削除する。
7. threshold `1/8`を明示的に変更できる引数として残し、ToyとMSTで保守性を確認する。

### 誤差の分離

次の2種類を混同しない。

1. voxel gridが実際のemissionを表現できない誤差
2. 表現済みの三線形補間profile `S f`を有限個のsubvoxel点で積分する誤差

source resを上げて改善できるのは2だけである。1を改善するにはvoxel grid自体を細かくする。
constant、linear、square、Gaussianを使う検証では、同じvoxel gridと同じ`S`を使い、quadrature
だけを変えて2を評価する。

## 2. MST実形状と境界の数値検証

### MST profile検証

まず `d=75 mm`程度の短時間モデルで、次のemission profileを投影する。

1. insideで一定、outsideでゼロ
2. 小半径方向に滑らかなGaussian
3. plasma境界付近まで値を持つprofile
4. 非対称なprofile

各profileについて、fixed resとadaptiveの総光量、pixel差、相対L1/L2誤差、最大pixel誤差、画像断面、
実行時間を比較する。detectorの `subpixel_resolution=1,2,5`について画像分布の収束を確認し、
rectangle Eyeとslit形状も個別に評価する。

短時間モデルで仕様を確定した後、約110万voxel、`d=25 mm`相当を最終benchmarkとして一度実行する。

### inside境界と未知変数

- inside voxelの三線形補間にoutside側の隣接voxel値が混ざる条件を可視化する。
- outside emissionをゼロ固定するか、projection matrixの列から除外するか、inverse problem側で
  拘束するかを仕様化する。
- 定数profileの総光量保存と、inside境界を横切るprofileの回帰テストを追加する。

## 3. Fine voxel gridへの対応

### 格子幅の比較

emissionの空間分解能と有限体積積分精度を分離して、少なくとも次を比較する。

- `d=25 mm`, source res 1または2
- `d=12.5 mm`, source res 1
- `d=10 mm`, source res 1

`d=10 mm`のbounding gridは約 `405 x 405 x 105`、約1720万voxelになる。projection時間だけでなく、
visibility配列、projection列数、cache、inverse problemを含むpeak memoryを測定する。

### Active voxel compact mapping

torus外、inside外、または全cameraから常に不可視なvoxelをprojection matrixの列から除外する。

- compact indexと元のvoxel indexの双方向mappingを保存する。
- emission入力をcompact化し、投影・transpose・inverse problem後に元のgridへ戻せるAPIを定義する。
- mapping自体、projection列、visibility、cacheを含む実メモリ削減量を測定する。
- constant emission、camera別visibility、pickle roundtripの回帰テストを追加する。

## 4. 未実装の物理拡張

### Eye内部位置ごとの局所visibility

wallやapertureによるvisibilityがEye内部位置で変化する場合を扱う。まず `0.5 x 4 mm`程度のslitと
実際のwall配置で中心visibilityとの差を定量化し、必要性が確認できた場合だけ実装する。

## 実装単位

1. **Geometry utility:** scalar-to-axis res変換、外接球指標、unit test
2. **Adaptive integration:** fully visible voxelの理想軸別res、bucket処理、診断値
3. **Toy acceptance:** 端点投影方式との精度・判定時間比較、`d`・`Z/f` sweepとpyplot
4. **MST acceptance:** `d=75 mm`でfixed/adaptive、profile、detector resを比較
5. **Cleanup:** 外接球方式の受け入れ後に端点投影方式を削除し、threshold `1/8`を文書化
6. **Boundary semantics:** inside/outside仕様と回帰テスト
7. **Compact mapping:** mapping API、cache、transpose、pickle、memory benchmark
8. **Fine-grid acceptance:** `d=10--25 mm`で時間・peak memory・投影誤差を比較
9. **Conditional physics:** slitでEye内部局所visibilityの必要性を判定

1--5を現在のbranchで完了させてから、6以降を独立したbranchへ分ける。
