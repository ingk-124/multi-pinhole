# プロジェクト概要

Projection matrix の検証項目と今後の改善候補は
[`projection-roadmap.md`](projection-roadmap.md) にまとめています。

## 目的

`multi_pinhole` は、プラズマの X 線 pinhole カメラ撮像をシミュレートするパッケージです（MST 磁場逆転ピンチ実験のために作られましたが、MST 固有の仕組みではありません）。ボクセルグリッド上に定義された3次元の発光分布と、1台以上のカメラ（それぞれが複数の pinhole または凹レンズの「eye」を持つ）が与えられたとき、ボクセル強度を検出器ピクセル強度へ写像する疎な線形演算子を計算します。この演算子こそがシミュレーションの本質的な出力です——一度これを手に入れれば、任意の発光分布の「画像をレンダリングする」ことは1回の疎行列・ベクトル積で済みますし、逆に画像からの逆問題（トモグラフィ再構成）も同じ行列に対する線形逆問題として扱えます。

このパッケージは、ワールド・カメラ・pinhole/eye・スクリーン（画像）という4つの座標系を軸に構成されており、これらは `multi_pinhole.core` の中で明確に定式化されています。このパッケージにおけるあらゆる幾何計算は、これらの座標系間の変換の組み合わせです。その変換の連鎖については `docs/core.md` で詳しく解説しています。

## 主要コンポーネント

- **コア光学系**（`multi_pinhole.core`）—— `Eye`（単一の pinhole／レンズチャンネル）、`Aperture`（解析形状または STL による遮蔽形状）、`Screen`（画素化された検出面とそのラスタライザ）、`Camera`（eye・aperture・screen をまとめ、ワールド空間に配置する）から構成されます。詳細は `docs/core.md` を参照してください。
- **ボクセルモデリング**（`multi_pinhole.voxel`）—— デカルト座標のボクセルグリッド（`Voxel`）と、トーラス状プラズマ発光を合成するためのヘルパー関数群です。詳細は下記「ボクセルグリッドの幾何」を参照してください。
- **座標変換**（`multi_pinhole.coordinates`）—— デカルト座標のボクセルグリッド点を円筒・トーラス・球座標などで**再解釈**するだけの純粋な関数群です。あくまでプロファイルを評価するための道具であり、グリッド自体は常にデカルト座標のままです。
- **ワールドの統括**（`multi_pinhole.world`）—— `World` は `Voxel`、1台以上の `Camera`、任意の STL 「壁」を1つのシーンにまとめます。eye ごとの可視性を計算し、ボクセル→スクリーンの投影行列を組み立てます。詳細は `docs/world.md` を参照してください。

## 典型的なワークフロー

1. **シーンを記述する。** `Voxel` グリッドを構築します——軸配列から直接作るか、等間隔なデカルト空間には `Voxel.uniform_voxel(ranges, shape)` を使います。光線を遮るべき STL の `walls` を必要に応じて読み込みます。
2. **光学系を設定する。** 1つ以上の `Eye`（pinhole の位置、焦点距離、aperture のサイズ／形状）を作成し、`Aperture` ジオメトリと組み合わせ、`Screen`（物理サイズ、ピクセルグリッド、サブピクセル分割）に取り付けます。
3. **`Camera` を組み立てる。** eye／aperture／screen から `Camera` を構成し、`camera_position` と回転によってワールド空間に配置します。
4. **`World` を構築する。** ボクセルグリッドとカメラから `World` を作り、`World.set_inside_vertices(...)` によって、どのボクセル頂点が対象体積の物理的な「内部」であるかをマークします（外部の頂点は以降の可視性・投影計算からスキップされます——これにより、例えば矩形のボクセル箱の中にあるトーラス形状のプラズマ体積を表現できます）。
5. **可視性と投影行列を計算する。** `World.set_projection_matrix()` は、各カメラの各 eye について、（aperture や壁によって遮られていない）可視なボクセルを判定し、疎な `(N_pixel, N_voxel)` 行列 `world.P_matrix[camera_idx]` を構築します。以下の具体例と `docs/world.md` の全パイプライン解説を参照してください。
6. **レンダリング、または逆問題を解く。** ボクセル強度ベクトル `emission`（形状 `(N_voxel,)`）が与えられれば、`world.P_matrix[camera_idx] @ emission` が全eyeを合算したピクセル画像になります。`world.projection[camera_idx][eye_idx] @ emission` は同じpixel座標における1つのeyeの寄与です。subpixelは面積積分の一時的な評価点であり、projection cacheには保持されません。

### 具体例：空の `World` から画像レンダリングまで

`examples/small_voxel_projection.py` を要点だけに絞ったものです。

```python
from multi_pinhole import Aperture, Camera, Eye, Screen, Voxel, World
import numpy as np

# 1. 各軸 [-3, 3] mm の 3x3x3 ボクセルグリッド。
voxel = Voxel.uniform_voxel(ranges=[[-3, 3], [-3, 3], [-3, 3]], shape=[3, 3, 3])

# 2-3. pinhole eye を1つ、円形 aperture を1つ、小さな screen を1つ組み合わせて Camera を構成。
camera = Camera(
    eyes=[Eye(eye_type="pinhole", eye_shape="circle", eye_size=1.0,
              focal_length=12.0, position=[0.0, 0.0])],
    apertures=Aperture(shape="circle", size=6.0, position=[0.0, 0.0, 25.0],
                        resolution=24, max_size=24.0),
    screen=Screen(screen_shape="rectangle", screen_size=[12.0, 12.0],
                  pixel_shape=(8, 8), subpixel_resolution=2),
    camera_position=[0.0, 0.0, -60.0],
)

# 4. ボクセルグリッドとカメラを World に束ね、すべての頂点を「内部」としてマーク。
world = World(voxel=voxel, cameras=[camera], verbose=0)
world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))

# 5. 可視性と、疎なボクセル→スクリーン投影行列を計算する。
world.set_projection_matrix(res=1, verbose=0, parallel=1)

# 6. レンダリング：ボクセルごとに発光値を決め、あとは疎行列積1回で画像を得る。
emission = np.exp(-((voxel.gravity_center[:, 0] / 2.2) ** 2
                    + (voxel.gravity_center[:, 1] / 1.8) ** 2
                    + (voxel.gravity_center[:, 2] / 2.6) ** 2))
pixel_image = world.P_matrix[0] @ emission      # 全eye、形状 (N_pixel,)
eye_image = world.projection[0][0] @ emission   # eye 0、形状 (N_pixel,)
```

内部的には、ステップ5（`set_projection_matrix`）がもっともコストの高い部分です。カメラの各 eye について、(a) すべてのボクセルの8個の角頂点をすべての aperture・壁に対して光線追跡し、不可視／部分的に可視／完全に可視に分類し、(b) 可視なボクセルについてサブボクセル点をサンプリングし、`Camera.calc_image_vec`（`docs/core.md` で説明する pinhole 投影＋ラスタライズのパイプライン）で eye を通して投影し、(c) それらのサブボクセルサンプルを積分してボクセルあたり1つの重みへ戻します。これらの各サブステップは `docs/world.md` で詳しく説明しています。

## ボクセルグリッドの幾何

`Voxel` は（必ずしも等間隔ではない）矩形格子の3次元グリッドであり、グリッド線の位置を表す3本の1次元軸配列 `x_axis`、`y_axis`、`z_axis` によって定義されます。この軸から、`Voxel.update()` が（ボクセルごとの Python ループを使わず）ベクトル化された方法ですべての派生量を導出します。

* **グリッド点**は3本の軸の直積 `(N_x+1) × (N_y+1) × (N_z+1)` であり、`z` を最も速く変化させ、その次に `y`、最後に `x` の順でフラット化されます（グリッド形状 `(N_x', N_y', N_z')` に対して線形インデックス `n = k + N_z'·(j + N_y'·i)`）。
* **ボクセル**は隣接するグリッド線の間の `N_x × N_y × N_z` 個のセルです。ボクセル `(i, j, k)` の8個の角頂点は、そのボクセルの基準となる線形グリッドインデックスに固定のオフセットパターン（`{0,1} × {0,1} × {0,1}` の組み合わせを、線形インデックスのオフセット `{0, 1, N_z', N_z'+1, N_z'·N_y', ...}` として表現したもの）を加えることで得られます——これは純粋なインデックス演算のトリックであり、呼び出し側が実際に `Voxel.vertices` を要求しない限り、明示的な `(N_voxel, 8, 3)` の頂点座標配列を構築せずに済みます。
* 各ボクセルの**体積**は3辺の長さの積（`dx · dy · dz`）であり、**重心**は8個の角の中点です——どちらも軸ごとに計算してブロードキャストされ、ボクセルごとに計算されるわけではありません。
* **サブボクセルサンプリング。** 補間／積分のために（`World` の投影パイプラインで多用されます。`docs/world.md` を参照）、1つのボクセルを `res = (x_res, y_res, z_res)` のサブボクセルサンプル点グリッドへ細分できます。`interpolate_matrix_from_vertices(res)` は三線形補間の重み行列を構築します。親ボクセル内の分数位置 `(a, b, c)`（`a, b, c ∈ [0, 1]`）にあるサブボクセルサンプル点は、ボクセルの8個の角頂点の値の重み付き和として表現され、重みは `(1−a)(1−b)(1−c)`、`(1−a)(1−b)c`、…、`abc` という標準的な三線形補間の基底になります。

### プロファイル評価のための座標変換

グリッド自体は常にデカルト座標です。`Voxel.normalized_coordinates()` は、デカルト座標の点（デフォルトではボクセルの重心）を任意で別の座標系に**再解釈**します。これはトーラス座標や円筒座標で書いたプロファイル関数を、その装置の対称性に自然な形で評価できるようにするためです。`multi_pinhole.coordinates` はそのような変換を5種類実装しており、いずれもデカルト座標 `(x, y, z)` を受け取って正規化座標を返します。

* **cartesian（デカルト）** —— 各軸を、その設定された半分の範囲でスケーリングするだけです。
* **cylindrical（円筒）** `(r, theta, z)` —— `r = sqrt(x²+y²)/a`、`theta = atan2(y, x)`、`z` は `h/2` でスケーリングされます。
* **torus（トーラス）** `(r, theta, phi)` —— 主半径 `R_0`、副半径 `a` のトーラスに対して：`R = sqrt(x²+y²)`、`r = sqrt((R−R_0)² + z²)/a`、`theta = atan2(z, R−R_0)`（poloidal 角、outboard 中間面で `0`）、`phi = atan2(−y, x)`（toroidal 角、`+z` 側から見て時計回りに増加）。`torus_inverse` は同じ構成で両方の角度の符号／基準を反転したもの（`theta` は inboard 中間面基準、`phi` は反時計回り）で、いずれも右手系の `(r, theta, phi)` です。
* **spherical（球）** `(r, theta, phi)` —— `distance = sqrt(x²+y²+z²)`、`r = distance/a`、`theta = arccos(z/distance)`（`+z` から測る `[0, pi]` の極角）、`phi = atan2(y, x)`（`+x` から反時計回りの `[-pi, pi]` の方位角）。reference radius `a` は `r` だけをscaleし、角度には影響しません。原点では `theta=nan`、z軸上の `phi` は数学的には未定義ですが NumPy `atan2` の結果に従います。

`multi_pinhole.profiles` は、これらの座標系の上で合成プロファイルを評価するための、組み合わせ可能なヘルパー関数を提供します。shifted polar 座標、kinked/flattened な径方向座標、そして torus 座標の `Voxel` から直接 profile 関数を評価する薄い wrapper を含みます。描画、フィッティング、実験固有の診断は core profile API の外側に置く想定です。

## 注目すべき機能

- カメラは複数の eye を同時にサポートし（マルチ pinhole 撮像）、各 eye は独立した位置・焦点距離・aperture の形状／サイズ・波長域を持てます。
- aperture は解析形状（円／楕円／矩形）または任意の STL メッシュを受け付け、硬い遮蔽物として扱われます。`check_visible` は eye と各候補点の間で2段階の可視性テスト（コーンによる事前フィルタ、その後の Möller–Trumbore 三角形交差判定）を実行します——詳しくは `docs/utilities.md` を参照してください。
- スクリーンのラスタライズ（`Screen.ray2image_grid`）は etendue で重み付けされた疎な CSR/CSC 行列を用いるため、数百万本の光線を密な配列を一切実体化せずにサブピクセル画像へ蓄積できます——詳しくは `docs/core.md` を参照してください。
- `World.set_projection_matrix` は、コストの高いサブボクセルのサンプリング・投影処理を `ThreadPoolExecutor` で並列化し、推定された疎度に基づいて適応的にチャンク分割することでメモリ使用量を抑えます——詳しくは `docs/world.md` を参照してください。

## 拡張性

このプロジェクトは、STL 処理（`multi_pinhole.utils.stl_utils`）、進捗表示に対応したロギング（`multi_pinhole.utils.my_stdio`）といったユーティリティ関数を主要クラスの下層に配置しています。そのため、新しい光学要素やカスタムのワークフローは、既存の座標変換・可視性判定・可視化ルーチンを再利用でき、内部のジオメトリ計算を一から実装し直す必要はありません。これらの構成要素の詳細は `docs/utilities.md` を参照してください。
