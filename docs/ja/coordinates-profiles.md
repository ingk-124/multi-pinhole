# 座標・profile・補間

> **Level 1のworkflowとLevel 2のreference。** 前半はCartesian voxelから
> plasma profileを作り検出器へ投影する通常手順です。後半の座標規約表と補間の
> 注意は、符号・normalization・数値的意味を確認する時に読みます。

このページは次の3操作の正本です。

1. Cartesian voxel点をplasmaに自然な座標系で解釈する
2. その座標上で再利用可能なscalar profileを評価する
3. voxel-center値をR–Z断面などの任意点へ補間する

これらの操作で`Voxel` grid自体が曲線座標へ変わることはありません。

## Voxelからemission profileを作る

profile関数はmoduleとしてimportします。

```python
from multi_pinhole import profiles
```

個々のprofile関数は`from multi_pinhole import *`には追加されません。
profileの基本入力は正規化poloidal Cartesian座標

$$
x=(R-R_0)/a,\qquad y=Z/a
$$

です。`+x`はR外向き、`+y`は上向きです。

```python
x, y, phi = voxel.to_coordinates(
    "poloidal_cartesian_inverse",
    normalized=True,
    major_radius=508,
    minor_radius=250,
).T

emission = profiles.axisymmetric_profile(
    x,
    y,
    A=1.0,
    delta=0.1,
    alpha=2.0,
    beta=3.0,
    edge_value=0.0,
)
```

`emission`は`(N_voxel,)`で、`voxel.gravity_center`と同じ順序です。そのまま
投影できます。

```python
image = world.project(emission, camera_idx="main")
```

発光領域外にはNaNではなく0を使います。NaNは疎行列積へ伝播します。

## profileの選び方

| 関数 | 用途 |
|---|---|
| `axisymmetric_profile` | 水平方向にshiftしたpoloidal対称profile |
| `kinked_profile` | 指定角へ半径依存変位するprofile |
| `flattening_profile` | kink profileへ全角度または局所的なdensity islandをblendするprofile |
| `crescent_profile` | kink mapにfoldがあれば解析的fold位置からclipするprofile |
| `helical_center_angle` | 参照位置のcenter angleを別toroidal位置へ伝播 |

評価座標`x`、`y`と`center_angle_xy`はNumPy規則でbroadcastします。`delta`、
`xi_0`、`rho_s`、`d`などのshape parameterは有限scalarです。optionalなprofile
制御parameterはkeyword-onlyです。`A`と`edge_value`には解析で定めた物理単位を
持たせられますが、座標とshape parameterは無次元です。

`center_angle_xy`はpoloidal外向き`+x`から上向き`+y`へ反時計回りに測ります。
toroidal `phi`の符号規約とは独立です。

helical構造なら次のように使います。

```python
center_angle_xy = profiles.helical_center_angle(
    phi,
    center_angle_xy_ref=0.2,
    m=1,
    n=-1,
    phi_ref=0.0,
)

emission = profiles.kinked_profile(
    x,
    y,
    A=1.0,
    delta=0.1,
    alpha=2.0,
    beta=3.0,
    xi_0=0.2,
    rho_s=0.5,
    d=2.0,
    center_angle_xy=center_angle_xy,
)
```

`helical_center_angle`は
`center_angle_xy_ref + (n/m) * (phi - phi_ref)`を計算し、角度をwrapしません。
`torus`か`torus_inverse`かは判定しないため、選んだ`phi`規約と整合する符号の
`n`を呼び出し側が渡します。

### static shiftとkinkのnormalization

`delta`は水平方向だけのstaticなShafranov shift相当の変位です。各rayで常に
normalizeされ、元の円形wallはstatic半径`rho=1`に保たれます。このstatic shift後、
kink前の半径を`rho_0`と書くと、kink変位は

$$
\xi(\rho_0)=\xi_0\exp\left[-\left(\frac{\rho_0}{\rho_s}\right)^d\right]
$$

で、`center_angle_xy`方向へ作用します。defaultの`normalize_kink=False`では、
kink後の半径を再normalizeせず、wallでkink変位が非zeroでも構いません。
wallでkink後の半径も1にしたい場合だけ、`kinked_*`、
`flattening_profile`に`normalize_kink=True`を指定します。

`crescent_*`には意図的に`normalize_kink` optionがありません。再normalizeしない
kink mapのfoldをLambert Wの`-1` branchで解析的に求め、その半径を基準に非単調
部分をflattenします。実数foldがなければkinked座標を変更せず返します。これは
fixed-wallの現象論的crescent modelで、`xi_0`は元の正規化Cartesian frameでの
変位です。

### flatteningの制御

`flattening_profile`は現象論的density-island modelであり、平衡や輸送を解く
solverではありません。基本動作は全角度flattening（`localized=False`）です。
flattening半径`rho_flat`を選び、省略時は`rho_s`を使います。kink前の半径`rho_0`と
kink半径からtarget island半径を構成します。

$$
\rho_{\mathrm{limit}}=\max(\rho_{\mathrm{flat}},\rho_0)
$$

$$
\rho_{\mathrm{island}}
=\min(\rho_{\mathrm{kink}},\rho_{\mathrm{limit}})
$$

通常のtwo-power profileを両方の半径で独立に評価します。

$$
n_{\mathrm{kink}}=n(\rho_{\mathrm{kink}})
$$

$$
n_{\mathrm{island}}=n(\rho_{\mathrm{island}})
$$

最終的には半径ではなくdensityをblendします。

$$
n_{\mathrm{flat}}
=(1-\lambda)n_{\mathrm{kink}}+\lambda n_{\mathrm{island}}
$$

したがって`lam_0`は閉区間`[0, 1]`の無次元density blend比です。defaultの
`localized=False`では全poloidal角で`lambda=lam_0`です。

```python
emission = profiles.flattening_profile(
    x,
    y,
    A=1.0,
    delta=0.1,
    alpha=2.0,
    beta=3.0,
    xi_0=0.2,
    rho_s=0.3,
    d=2.0,
    lam_0=1.0,
)
```

flattening半径をkink減衰半径`rho_s`から独立させる時だけ`rho_flat`を指定します。
localization専用引数はこのmodeでは無視されます。

`localized=True`はdensity blendを半径・角度方向に制限するoptional extensionです。
この場合は正の幅`w`が必須で、

$$
\lambda=\lambda_0 G(\rho_{\mathrm{kink}};\rho_{\mathrm{flat}},w)
\frac{1+\cos(\theta')}{2}
$$

$$
G(\rho;\rho_{\mathrm{flat}},w)
=\exp\left[-\left|\frac{2(\rho-\rho_{\mathrm{flat}})}{w}\right|^d\right]
$$

です。`theta'`は`center_angle_xy + flattening_angle_offset`を基準とするkink後の
角度です。optionalなdistortionは
`theta' = Delta theta + gamma*sin(Delta theta)`です。`w`はfull e-folding widthで、
`|rho-rho_flat|=w/2`において`G=exp(-1)`です。Gaussianの標準偏差ではありません。
`d=2`なら`sigma=w/(2*sqrt(2))`、一般の`d`でfull width at half maximumは
`w*(ln(2))**(1/d)`です。`blend_edge=None`では`rho=0`と`rho=1`でGaussianを
抑制せず、正の`blend_edge`を指定するとtaperを有効にします。

```python
localized_emission = profiles.flattening_profile(
    x,
    y,
    A=1.0,
    delta=0.1,
    alpha=2.0,
    beta=3.0,
    xi_0=0.2,
    rho_s=0.3,
    d=2.0,
    localized=True,
    rho_flat=0.4,
    w=0.2,
)
```

`smooth_eps=0`では`rho_island`の構成にNumPyの厳密なmaximum/minimumを使います。
正の値を指定すると両演算を同じscaleでsmooth化します。

`edge_value`はeffective半径`rho=1`におけるtwo-power profile値です。
`normalize_kink=False`では物理的な円形wall上のkink半径が角度依存になるため、
単一の`edge_value`ではwall値が一様になることを保証しません。wall境界条件を、
wallで非zeroのkink変位を許すことより優先する場合は`normalize_kink=True`を使います。
どちらの場合も元のunit disk外は0です。

parameter比較は次の実行例にあります。

```bash
python examples/profiles_demo.py
```

## 任意点の座標変換

`Voxel.to_coordinates`は、Voxelに保存されたlegacy `coordinate_type`を変更せず、
Cartesian点を変換します。

```python
coordinates = voxel.to_coordinates(
    "cylindrical",
    points=[[1.0, 2.0, 3.0]],
)
```

`points`には`"centers"`、`"vertices"`、または`(..., 3)`配列を指定できます。
逆変換は名前付き成分を受け取り、NumPy broadcastを行います。

```python
xyz = voxel.from_coordinates(
    "cylindrical",
    R=np.linspace(1, 2, 5)[:, None],
    phi=np.linspace(0, 2 * np.pi, 100)[None, :],
    Z=0,
)
```

singleton軸を付けることで`(5, 100)`のR–phi meshになります。独立なshape
`(5,)`と`(100,)`はそのままではbroadcastできません。

`normalized_coordinates()`はVoxelに設定済みの座標を使う互換APIです。新しい
解析codeでは、規約とscaleが呼び出し箇所に見える`to_coordinates(...)`を
推奨します。

## 座標規約

| `coordinate_type` | 成分 | 角度規約 | geometry / normalization parameter |
|---|---|---|---|
| `cartesian` | `x, y, z` | — | `width`, `depth`, `height`。各half-rangeで割る |
| `cylindrical` | `R, phi, Z` | `phi=atan2(y,x)`。+zから見て+xより反時計回り | `radius`, `height`。Zは`height/2`で割る |
| `torus` | `r, theta, phi` | theta=0はoutboard、上向き正。phiは時計回り | `major_radius`, `minor_radius` |
| `torus_inverse` | `r, theta, phi` | theta=0はinboard、上向き正。phiは反時計回り | `major_radius`, `minor_radius` |
| `poloidal_cartesian` | `x, y, phi` | `x=R-R0`, `y=Z`。phiは時計回り | `major_radius`, `minor_radius` |
| `poloidal_cartesian_inverse` | `x, y, phi` | poloidal軸は同じ。phiは反時計回り | `major_radius`, `minor_radius` |
| `spherical` | `r, theta, phi` | thetaは+zからの極角。phiは反時計回り | `radius` |

`normalized=False`ならradial/axial成分は利用者の長さ単位を保ちます。角度は常に
radです。`normalized=True`では必要scaleをすべて明示し、省略するとerrorです。

## voxel-center値の補間

`center_interpolator`は3本のvoxel-center軸に対するSciPy
`RegularGridInterpolator`を構築します。

```python
interpolate = voxel.center_interpolator(
    emission,
    method="linear",
    bounds_error=False,
    fill_value=np.nan,
)

values = interpolate(points=[[500.0, 0.0, 20.0]])
```

fieldは`voxel.shape`、`(N_voxel,)`、または末尾にvector/tensor次元を持つshapeを
受け付けます。Cartesian queryは`(..., 3)`です。代わりに座標系と名前付き成分を
指定することもできます。

### 固定phiのR–Z断面

物理的なR–Z平面にはcylindrical成分を使います。cylindrical `phi`は常にworld
`+x`から、`+z`側から見て反時計回りです。

```python
import matplotlib.pyplot as plt
import numpy as np

interpolate = voxel.center_interpolator(
    emission,
    method="linear",
    bounds_error=False,
    fill_value=np.nan,
)

R = np.linspace(250, 750, 201)
Z = np.linspace(-250, 250, 201)
RR, ZZ = np.meshgrid(R, Z, indexing="xy")

section = interpolate(
    coordinate_type="cylindrical",
    R=RR,
    phi=np.deg2rad(45),
    Z=ZZ,
)

fig, ax = plt.subplots()
mesh = ax.pcolormesh(RR, ZZ, section, shading="auto")
ax.set_aspect("equal")
ax.set_xlabel("R [mm]")
ax.set_ylabel("Z [mm]")
fig.colorbar(mesh, ax=ax, label="Emission")
plt.show()
```

これはvoxel centerでsample済みの値を補間したもので、体積積分ではなく、元fieldの
物理解像度を高める処理でもありません。解析profile関数が残っているなら、粗い
voxel値を補間するより表示grid上で関数を直接評価する方が正確です。

`plot_voxel_slice`は別用途です。既存のCartesian voxel-center面をcell境界付きで
表示し、補間は行いません。詳しくは[可視化](visualization.md)を参照してください。
