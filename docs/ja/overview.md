# 概要と最初の投影

> **対象:** 初めて使う学部生から、既存のcamera geometryで解析する研究者まで。
> このページでは内部のray–triangle判定や疎行列組立を説明しません。
> 数値モデルの根拠が必要になった時だけ[Level 2](README.md#level-2--科学モデルを理解する)
> へ進んでください。

## 何を計算するライブラリか

`multi_pinhole`は、3次元のvoxel発光分布をpinhole cameraの検出器画像へ写す
libraryです。geometryから疎な投影行列

$$
\mathbf{g}=\mathbf{P}\mathbf{f}
$$

を一度作ります。$\mathbf{f}$はvoxelごとの発光、$\mathbf{g}$はpixel信号です。
同じgeometryなら発光分布を変えてもray tracingをやり直さず、`World.project`
による行列積だけで画像を作れます。

最初は次の5つだけ区別できれば十分です。

| object | 意味 |
|---|---|
| `Voxel` | 発光分布を置くCartesian cell |
| `Eye` | 1つのpinholeまたは有限開口channel |
| `Aperture` | 光を通す穴と周囲の遮光形状 |
| `Screen` | pixel化された検出面 |
| `Camera` / `World` | 光学系 / scene全体と投影cache |

## 推奨workflow

### 1. JSONでsceneを定義する

再現可能な解析では、Python内でobjectを個別に組み立てるよりJSONを推奨します。

```python
from multi_pinhole import World

world = World.from_config("scene.json")
```

JSONにはvoxel範囲、camera位置・向き、eye、screen、aperture、必要ならwallを
記述します。詳しい組み方は[World JSON設定ガイド](world-config-guide.md)に
集約しています。全keyの型を調べる場合だけ[JSON schema](config.md)を参照します。

### 2. 高価な計算の前にgeometryを確認する

- cameraがplasmaを向いているか
- wallのportを視線が通っているか
- 長さの単位がscene全体で統一されているか
- `inside`が発光させたい領域を覆っているか

を確認します。`inside`は発光計算対象を選ぶmaskで、光を遮るwallではありません。

```python
world.find_visible_voxels("main", verbose=1)
work = world.preflight_projection(res=3, partial_res=3)
print(work.summary())
```

`visible_voxels`の状態は、`0=不可視`, `1=部分可視`, `2=完全可視`です。
`preflight_projection`は投影行列をまだ作らず、必要sample数の見積りを返します。

### 3. 投影行列を作る

```python
world.set_projection_matrix(
    res=3,
    partial_res=3,
    parallel=4,
    verbose=1,
)
```

`res`を大きくするとsource積分は細かくなりますが、計算量も増えます。
値は精度保証ではないため、最終解析では`res`を変えた収束確認が必要です。
wallやaperture境界を横切る部分可視voxelでは、特に`partial_res`が効きます。

### 4. emissionを投影する

```python
import numpy as np

x, y, z = world.voxel.gravity_center.T
emission = np.exp(-((x / 100) ** 2 + (y / 100) ** 2 + (z / 150) ** 2))

image = world.project(emission, camera_idx="main")
world.cameras["main"].screen.show_image(image)
```

`emission`は`(N_voxel,)`、複数時刻なら`(N_voxel, N_time)`です。非有限値は
行列積へ伝播するため、計算対象外を表す値にはNaNではなく0を使います。

voxel profileの3D表示と断面表示は[可視化ガイド](visualization.md)にあります。

### 5. 必要なら計算済みWorldを保存する

```python
world.save("checkpoint.mpw")
```

JSONはscene入力だけ、`.mpw`はvisibilityとprojection cacheを含むcheckpointです。
使い分けとsecurity上の注意は[serialization](serialization.md)にまとめています。

## 座標について最低限知ること

- voxel gridとworldはCartesian `(x, y, z)`です。
- camera内部にはcamera/eye/screen座標がありますが、通常利用では変換をlibraryに
  任せます。
- plasma profileの評価時だけ、同じCartesian点をcylindrical、torus、
  poloidal Cartesian等へ変換できます。grid自体が曲線座標になるわけではありません。
- angleの符号と基準は解析結果を左右します。使用する座標系は
  `Voxel.to_coordinates()`のdocstringと
  [optics・座標の説明](core.md#利用者が知るべき座標系)で確認してください。

## よくある間違い

- **単位の混在:** libraryはmmとmを判別しません。
- **wallとinsideの混同:** wallは遮蔽、insideはsource領域です。
- **NaNを不可視voxelに入れる:** `P @ emission`全体へNaNが伝播し得ます。
- **pixel配列を通常画像と同じ向きだと思う:** `Screen.show_image`を優先します。
- **1つのresolutionだけで精度を判断する:** 最終値は収束確認します。
- **JSONだけでcacheも保存されると思う:** cache保存は`.mpw`です。

> **通常利用はここまでで十分です。**
> 以下のページは必要になった目的に応じて選んでください。

## 次に読むページ

- cameraをJSONで組む → [World JSON設定ガイド](world-config-guide.md)
- plasma profileを作る、R–Z断面を補間する →
  [座標・profile・補間](coordinates-profiles.md)
- 結果を描く → [可視化](visualization.md)
- pinhole式、finite-eye、detector積分を理解する → [core](core.md)
- visibility、source積分、近似精度を調べる → [world](world.md)
- 正確なJSON keyを引く → [config reference](config.md)
