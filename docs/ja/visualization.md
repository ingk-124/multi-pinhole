# 可視化

> **Level 1 — 作業ガイド。** voxel emissionと検出器dataの描画方法はこの
> ページを正本とします。Plotly、Matplotlib、meshの内部実装は
> [utilities](utilities.md)へ分離しています。

## cameraとwallの配置確認

visibilityや投影行列を計算する前にgeometryを確認します。最も高水準な
Matplotlib表示は次です。

```python
import matplotlib.pyplot as plt

ax = world.draw_camera_orientation(
    show_fig=False,
    elev=60,
    azim=-30,
    facecolors="lightgray",
    alpha=0.15,
)
plt.show()
```

名前は`draw_camera_orientation`ですが、`World`版は配置確認に必要なscene全体を
表示します。

- voxelの境界範囲
- 全cameraの位置とlocal X/Y/Z軸
- 登録済みの全wall mesh
- voxel、wall、cameraを含むよう調整したaxis limit

各cameraのeye、aperture、screenを実寸で描く関数ではありません。それらは
camera local座標で確認します。

```python
camera = world.cameras["main"]
ax = camera.draw_optical_system(
    show_focal_length=True,
    show_aperture=True,
    show_screen=True,
)
plt.show()
```

`Camera.draw_optical_system`はeye位置、focal length、aperture support mesh、
screen配置の確認に使います。`Camera.draw_camera_orientation()`は選択したcameraの
world位置と軸だけを表示します。

wallとcamera軸を対話的に確認する場合はPlotly helperを合成します。

```python
import plotly.graph_objects as go
from multi_pinhole.utils import stl_utils

fig = go.Figure()
for wall in world.walls:
    stl_utils.plotly_show_stl(
        wall,
        fig=fig,
        color="lightgray",
        opacity=0.2,
        show_edges=False,
        show_fig=False,
    )
for camera in world.cameras.values():
    camera.draw_camera_orientation_plotly(
        fig,
        axis_length=100,
        show_fig=False,
    )
fig.show()
```

wallまたはcustom aperture STLだけを調べる場合は
`stl_utils.show_stl(model)`または`stl_utils.plotly_show_stl(model)`を使います。

これらは設定済みgeometryを描くだけで、光線がapertureやwall portを通ることを
保証しません。目視確認後に`world.find_visible_voxels(camera_idx)`を実行し、
高価な投影計算の前に0/1/2のvisibility stateを確認してください。

## voxel emissionの3D表示

`plot_voxel_volume`は`Voxel`とemissionを一緒に受け取り、shapeを検証し、
正しいflatten順で`voxel.gravity_center`を使います。

```python
import numpy as np
from multi_pinhole import Voxel
from multi_pinhole.utils.plot import plot_voxel_volume

voxel = Voxel.uniform_voxel(
    ranges=[[-1, 1], [-1, 1], [-1, 1]],
    shape=[30, 30, 30],
)
x, y, z = voxel.gravity_center.T
emission = np.exp(-4 * (x**2 + y**2 + z**2))

fig = plot_voxel_volume(
    voxel,
    emission,
    length_unit="m",
    value_label="Emissivity [W m⁻³]",
    opacity=0.15,
    surface_count=20,
    colorscale="Viridis",
)
fig.show()
```

`emission`は`(N_voxel,)`または`voxel.shape`を受け付け、非有限値は描画から
除外します。少なくとも1つのeyeから見えるvoxelだけなら
`mask=world.visible_voxels["camera"].any(axis=0)`を指定できます。ただし
`World.project`へ渡す発光値では不可視領域をNaNではなく0にしてください。
NaNは疎行列積の出力へ伝播します。

## 軸に平行な断面

```python
import matplotlib.pyplot as plt
from multi_pinhole.utils.plot import plot_voxel_slice

plot_voxel_slice(
    voxel,
    emission,
    axis="z",
    coordinate=0,
    length_unit="m",
    colorbar_label="Emissivity [W m⁻³]",
)
plt.show()
```

`coordinate`から最寄りのvoxel-center面を選びます。面indexを厳密に指定するなら
`index=`を使います。voxel境界軸を`pcolormesh`へ渡すため、点sampleではなく
cellの物理的な広がりとして表示されます。

このhelperは補間しません。`Voxel.center_interpolator`を使う滑らかな固定phi
R–Z断面は[座標・profile・補間](coordinates-profiles.md#固定phiのrz断面)を
参照してください。

## 検出器画像

```python
image = world.project(emission, camera_idx="main")
world.cameras["main"].screen.show_image(image)
```

screen画像はscreenの`(u, v)`規約に従います。pixel順序と物理extentを正しく
処理する`Screen.show_image`を優先してください。手動で`pcolormesh`する場合は
screen座標配列を用い、通常の画像row順と同じだと仮定せず、縦軸の向きを明示的に
選びます。

一連の投影と両可視化helperは
[`examples/example.py`](../../examples/example.py)、投影行列を作る前のvisibility
確認は[`examples/relax`](../../examples/relax/README.md)にあります。
