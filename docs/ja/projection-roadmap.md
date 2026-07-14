# Projection matrix roadmap

この文書には、projection matrixの高精度化・高速化について今後実施する項目だけを記録する。
完了した変更と撤回した設計はGit履歴で追跡し、ここには残さない。

## 1. Source quadratureの単純化

### 目的

fixed `res`を高精度計算と検証のreferenceとして維持し、adaptiveは明らかに点光源とみなせる
fully visible voxelだけを `res=(1, 1, 1)`へ落とす高速化optionとする。幾何判定を任意のemission
profileに対する誤差保証とは扱わない。

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

投影方向を特定しないscalar worst-caseなので、小さい方のreference scaleを採用する。`rho`が単一の
閾値以下なら `res=(1, 1, 1)`、それ以外は指定されたfixed resを使用する。閾値を小さくすると
保守的になり、細かいres側へ倒れる。

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
7. ToyとMSTの結果から保守的なdefault thresholdを決定し、明示的に変更できる引数として残す。

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

## 実施順序

1. 外接球によるres=1判定とscalar-to-axis res変換を実装
2. Toy sweepで端点投影方式との精度・判定時間を比較
3. `d=75 mm` MSTでfixed/adaptive、profile、detector resを比較
4. inside境界とoutside未知変数の仕様を確定
5. active voxel compact mappingを実装
6. `d=10--25 mm`のfine gridで時間・memory・投影誤差を比較
7. slitでEye内部局所visibilityの必要性を判定
