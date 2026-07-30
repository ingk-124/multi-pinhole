# World JSON設定の組み立て方

> **Level 1 — 作業ガイド。** sceneを作る時は上から順に読みます。判断基準と
> 典型的な失敗を説明し、全keyは重複掲載しません。網羅的な仕様は
> [schema reference](config.md)を引いてください。

このページでは、`World.from_config()` で読み込める、cacheを含まない可搬な
scene JSONを段階的に組み立てます。全keyと検証規則は
[schema reference](config.md)を参照してください。

## 1. voxel gridを定義する

ライブラリは数値に単位を付けないため、1ファイル内では同じ長さ単位を使います。
以下のfragmentはmmです。まずrootの`voxel` fieldを作ります。

```json
"voxel": {
  "type": "uniform",
  "ranges": [[250, 750], [-250, 250], [-250, 250]],
  "shape": [40, 40, 40],
  "coordinate": {
    "type": "torus",
    "parameters": {
      "major_radius": 508,
      "minor_radius": 250
    }
  },
  "sub_voxel_resolution": [1, 1, 1]
}
```

`ranges` はCartesian `x`, `y`, `z`の境界、`shape`はvoxel数
`(N_x, N_y, N_z)`です。grid vertex数ではありません。メモリと投影時間は
おおむね3成分の積に比例します。非等間隔gridではcompactな
`type: "uniform"`、`ranges`、`shape`の代わりに、次のようにexplicit boundaryを
指定します。

```json
"type": "axes",
"axes": {
  "x": [250, 300, 380, 500, 750],
  "y": [-250, -100, 0, 100, 250],
  "z": [-250, -50, 50, 250]
}
```

2形式は混在できません。`to_config()`は3軸が等間隔ならcompact形式を自動的に
使います。

物理的なプラズマ領域だけを残すには`inside`を追加します。

```json
"inside": {
  "type": "torus",
  "parameters": {
    "major_radius": 508,
    "minor_radius": 250
  }
}
```

voxel vertexで
`(sqrt(x^2 + y^2) - major_radius)^2 + z^2 <= minor_radius^2`
を評価します。これは発光領域の定義であり、光線を遮るwallではありません。

## 2. cameraの位置と向きを決める

既知の点へcameraを向ける指定が最も間違いにくい方法です。

```json
"cameras": [
  {
    "key": "main",
    "name": "equatorial camera",
    "position": [671.75, 671.75, 0],
    "orientation": {
      "look_point": [359.21, 359.21, 0],
      "right_point": [672.46, 671.04, 0]
    },
    "eyes": [{
      "type": "pinhole",
      "position": [0, 0],
      "focal_length": 20,
      "shape": "circle",
      "size": [0.25, 0.25],
      "wavelength_range": [0.01, 0.1]
    }],
    "screen": {
      "shape": "rectangle",
      "size": [8, 8],
      "pixel_shape": [32, 32],
      "subpixel_resolution": 3
    },
    "apertures": []
  }
]
```

`position`, `look_point`, `right_point`はworld絶対座標です。
`right_point - position`がscreen右方向を定め、視線方向と平行にはできません。
検出器の下方向が分かりやすい場合は`right_point`の代わりに`down_point`を
使えます。上級者は3×3の`rotation_matrix`も指定できますが、両方式を混在
させてはいけません。

eyeとapertureのpositionはcamera local座標です。重い投影計算の前に
向きを目視確認します。

```python
import plotly.graph_objects as go
from multi_pinhole import World

world = World.from_config("scene.json")
fig = go.Figure()
world.cameras["main"].draw_camera_orientation_plotly(fig, show_fig=False)
fig.show()
```

## 3. apertureを追加する

解析形状は開口と、その周囲の不透明meshを生成します。

```json
"apertures": [{
  "type": "analytic",
  "shape": "circle",
  "size": [10, 10],
  "position": [0, 0, 80],
  "direction": [0, 0, 1],
  "resolution": 40,
  "max_size": [200, 200]
}]
```

`resolution`は開口境界のsample数です。曲線の精度を上げるとvisibility計算も
重くなります。`max_size`は穴の直径ではなく、周囲の遮光板の広がりです。
関係する光線を覆える値にします。

CAD等の形状にはSTLを使えます。

```json
"apertures": [{
  "type": "stl",
  "path": "geometry/aperture.stl",
  "position": [0, 0, 80]
}]
```

相対pathはJSONファイル基準で解決されます。

## 4. 必要ならwallを追加する

```json
"walls": [{
  "type": "stl",
  "path": "geometry/vessel.stl"
}]
```

wallとaperture meshは光線を遮ります。穴のない閉じた容器を外から見るcameraは
何も見えません。STLに意図したportがあり、cameraがそこを向いているか確認して
ください。`wall`を省略するとcamera apertureだけがvisibilityを制限します。

## 5. 読み込み、確認、計算

```python
from multi_pinhole import World

world = World.from_config("scene.json")
world.find_visible_voxels("main", verbose=1)
world.set_projection_matrix(res=3, parallel=4, verbose=1)
```

JSONは再現可能なscene入力だけを保存し、visibility/projection cacheは保存
しません。計算済みcacheごと保存する場合はversion付き`.mpw` archiveを使います。
[serialization](serialization.md)も参照してください。

完全なRELAX設定と確認scriptは
[`examples/relax`](../../examples/relax/README.md)、voxel profileと検出器画像の
描画方法は[可視化ガイド](visualization.md)にあります。
