# Core モジュールリファレンス

## Detector quadrature の契約

`Screen.ray2image_grid` は有限 Eye の spot を detector cell へ積分します。
ellipse と cell の重なり**面積**は解析的に求めるため、小さい spot が cell
中心を外れただけで消えることはありません。一方、局所密度は rectangle
で 2×2 Gauss、1 cell 内に完全に入る ellipse で radial 2 × angular 8、
clipped ellipse の境界 cell で 4×4 masked midpoint により評価します。
最後の局所平均は近似で厳密な誤差保証はありません。subpixel refinement
は局所 PSF 精度に影響し、screen 境界で切れた spot は検出総量が減ります。
有限 Eye では detector 点を Eye 上へ戻す Jacobian と source/Eye 位置に
依存する局所立体角正規化密度を用います。

このドキュメントは `multi_pinhole.core` から再エクスポートされるクラスを説明します。それも、単なる API の形ではなく、それらが**何をどう計算しているか**——座標系の変換規則、pinhole 投影の式、光線をサブピクセルの重みへ変換するラスタライズアルゴリズム、aperture による遮蔽判定——を中心に説明します。

実装は責務別に、`multi_pinhole.eye`（Eye と ray 生成）、`multi_pinhole.aperture`（aperture／STL geometry）、`multi_pinhole.screen`（detector overlap、etendue quadrature、rasterizer）、`multi_pinhole.camera`（構成、座標変換、姿勢、描画）へ分割されています。`multi_pinhole.core` は後方互換 facade であり、旧import pathと新moduleから得られるclassは同一objectです。

## 4つの座標系

コア光学moduleの計算は、4つの座標系の間で点を移動させています。この連鎖を理解することが、本ドキュメントの残りを読み解く鍵になります。

```
world (x, y, z)  →  camera (X, Y, Z)  →  eye/pinhole (X', Y', Z')  →  screen/image (u, v)
```

1. **ワールド座標** `(x, y, z)` —— 唯一のグローバルなデカルト座標系です。`World` 内のすべて（ボクセル、カメラ、壁）はここで表現されます。プラズマプロファイルを評価するために円筒座標やトーラス座標として**再解釈**することもできますが（`multi_pinhole.coordinates` と `docs/overview.md` を参照）、内部表現は常にデカルト座標 `(x, y, z)` のままです。

2. **カメラ座標** `(X, Y, Z)` —— 1台の `Camera` に付随する右手系のデカルト座標系です。原点はそのカメラの screen の中心、`Z` は主光軸（screen から外向き）、`X` は右方向、`Y` は下方向を向きます。ワールド座標の点は、カメラ位置への平行移動とカメラの回転行列の適用によってカメラ座標へ変換されます——これがまさに `Camera.world2camera` です：`X_cam = R · (x_world − camera_position)`、実装は
   `(self.rotation_matrix @ (points - self.camera_position[None, :]).T).T`。

3. **eye（pinhole）座標** `(X', Y', Z')` —— カメラに搭載された `Eye` ごとに1つずつ存在します。1台のカメラは複数の eye を持つことができ、それぞれがカメラ平面内で `(X_h, Y_h)` だけオフセットし、焦点距離 `f` だけ screen から離れています。eye 座標系は、原点が eye の位置に来るようにカメラ座標系を平行移動しただけのものです：`X' = X − X_h`、`Y' = Y − Y_h`、`Z' = Z − f`（pinhole の場合。凹レンズの場合は `Z' = Z`。後述）。この変換が `Eye.camera2eye` です。

4. **画像座標** `(u, v)` —— screen 面上の2次元座標系で、原点は screen の**左上**の角です（中心ではありません）。`u` は下方向（カメラの `Y` に対応）、`v` は右方向（カメラの `X` に対応）に増加します。`(X, Y)` に対して軸の順序が入れ替わっている点に注意してください。screen の中心、すなわちカメラ座標系の原点は `(u_c, v_c) = screen_size / 2` にあります。`Screen.xy2uv` はまさにこの反転とシフトを行います：`uv = xy[..., ::-1] + screen_size / 2`。

2つの eye タイプは、pinhole 座標系が screen に対してどう配置されるかを変えます。

* **pinhole**（`eye_type="pinhole"`、`focal_length > 0`）：「eye」そのものが pinhole であり、`Z` 方向に焦点距離だけ screen からオフセットしています。位置と principal point はどちらも `(X_h, Y_h, f)` です。
* **凹レンズ**（`eye_type="concave_lens"`、`focal_length < 0`）：screen がレンズ面と一致するため、eye の位置は `(X_h, Y_h, 0)` になります。一方、投影式（後述）で使われる principal point は `(X_h, Y_h, f)` にオフセットされます。

これは `Eye.__init__` で強制されており、同時に `eye_size` を `(height, width)` のペアに正規化し、`eye_shape` を検証します。

## Rays

`Rays` は `multi_pinhole.rays` で定義され、`multi_pinhole.core`／`multi_pinhole` から再エクスポートされる immutable な dataclass で、1つの eye を通してシーンの点を投影した結果の幾何情報を保持します。各入力点について次を記録します。

* `Z`：eye 座標系での eye から点までの軸方向距離（符号付き。手前向き／奥向きのサンプルを区別できます）。
* `XY`：screen 上の投影先の位置。カメラの `(X, Y)` 座標系で表現され、eye の後方にある、あるいは何らかの理由で除外されたサンプルは `NaN` になります。これにより下流のコードは別途インデックスのリストを保持せず `np.isnan` でマスクできます。
* `zoom_rate`：拡大率 `1 + f / Z`（下記の投影の導出を参照）。eye の物理的な足跡を screen 上に描く際に拡大縮小するために必要な量です。
* `front_and_visible`：「eye の前方にある」（`Z > 0`）という条件と、外部から渡された aperture 遮蔽判定の結果を組み合わせた真偽値マスクです。

4つの配列はすべて対応しており、ファンシーインデックス（`Rays.__getitem__`）に対応しているため、呼び出し側は1つの真偽値マスクで4つのフィールドすべてをまとめてスライスできます。`Rays.n` と `Rays.n_visible` は光線の総数と生き残った光線数を公開しており、疎バッファのサイズ決定や空バッチのスキップを、別途 `np.count_nonzero` を呼ぶことなく行えます。

`Rays` インスタンスは `Eye.calc_rays`（これを生成する側）と `Screen` のラスタライザ（これを消費する側）の間に位置します——純粋なデータであり、生成元の `Eye` や `Camera` への参照は持ちません。

## Eye：pinhole 投影

`Eye` は、すでにカメラ座標系で表現された3次元の点を、screen 上の2次元の着地点へ変換します。`Eye.calc_rays` はこれを次の4ステップで行います（ドキュメント文字列自身の要約に対応します）。

1. **eye 座標系へ平行移動する。** `camera2eye` が eye のカメラ座標系での位置を引きます：`points_in_eye = points_in_camera − eye.position`。
2. **各点を分類する。** ある点が画像に寄与するのは、eye の前方にあり（`Z = points_in_eye[:, 2] > 0`、`front_only=True` の場合）、かつ外部から与えられた可視性マスク（aperture の遮蔽判定。後述）を通過した場合に限られます。この組み合わせたマスクが `front_and_visible` です。
3. **screen 上に投影する。** 生き残ったすべての点に対して、古典的な pinhole カメラの関係式がカメラの `(X, Y)` 単位で適用されます。

   ```
   XY = −(X', Y') / Z' · f + principal_point[:2]
   ```

   つまり、eye 座標系での横方向オフセットを（符号付きの）奥行きで割り、焦点距離でスケーリングし、符号を反転させ（pinhole の像は反転するため）、eye の principal point を中心に再配置します。具体的には、`position=(5, 0)`、`focal_length=20` で作った pinhole eye（`Eye.__init__` により `eye.position = eye.principal_point = (5, 0, 20)` になる）について、eye 座標系で `(X', Y', Z') = (2, 0, 10)` に位置する光源点（——これは例示のための数値であり、ソースコードから引用したものではありません。ソースコードには `calc_rays` 自体の数値例はありません——）は、`XY = −(2, 0)/10 · 20 + (5, 0) = (5 − 4, 0) = (1, 0)` に写像されます。
4. **zoom rate を計算する。** `zoom_rate = 1 + f / Z'`。これは、その奥行きにおける eye 自身の aperture の足跡の拡大率です：有限サイズの pinhole／レンズは、焦点面上の点を除けば点光源を完全な点には結像できません。そのため焦点面以外の奥行き `Z'` では、光源は screen 上でぼやけたスポットになり、そのサイズは eye の物理的な aperture サイズを `zoom_rate` でスケーリングしたものになります（後述の `Screen.ray2image_grid` を参照）。`Z' = 0` の点は前方判定のテストで除外されるため、生き残った点についてゼロ除算が起きることはありません。

`Eye.camera2eye` はステップ1のためのベクトル化された構成要素であり、それ以降はすべて `calc_rays` にインラインで記述されています。

## Aperture：遮蔽形状

`Aperture` は eye に届く光を制限する物理的な開口部を表します。解析形状（円／楕円／矩形）または明示的な STL メッシュのいずれかを受け付けます。解析形状の場合、`Aperture.set_model` が必要に応じて `stl_utils.generate_aperture_stl`（形状内部の Delaunay 三角形分割——メッシュがどう構築されるかは `docs/utilities.md` を参照）で STL メッシュを構築し、aperture の位置へ平行移動します。

この STL メッシュは飾りではなく、遮蔽判定に実際に使われるジオメトリです。`Camera` が点を投影するとき（`calc_image_vec`、後述）、カメラ上のすべての aperture について `stl_utils.check_visible(mesh_obj=aperture.stl_model, start=eye.position, grid_points=points_in_camera, ...)` を呼び出し、点が**すべての** aperture のメッシュをクリアした場合にのみ可視とします。つまり、**aperture は光を遮る面として扱われます**：STL メッシュは開口部の周囲にある不透明な材質であり、`check_visible` は eye と各候補点の間のレイ・メッシュ交差判定（Möller–Trumbore、コーンによる事前フィルタ付き——`docs/utilities.md` を参照）です。ある点が生き残るのは、eye からその点までの線分がメッシュを**横切らない**場合だけです。

## Screen：ピクセル／サブピクセルの幾何とラスタライズ

`Screen` は検出面を表します。`Screen.__init__` は物理的な `screen_shape`／`screen_size` を検証し、その矩形に `pixel_shape = (U_p, V_p)` のピクセルグリッドを敷き詰め、各ピクセルの中心を `positions()`（各軸に沿った単純な `linspace` を、ピクセル半分だけオフセットして、中心がセルの端ではなく真ん中に来るようにしたもの）によって計算します。`subpixel_resolution = k` を設定すると、各ピクセルが `k × k` のより細かいサブグリッドに分割され（`_set_variables`）、各ピクセルの `k²` 個のサブピクセルを合算する疎な `transform_matrix` が構築されます——これが `Screen.subpixel_to_pixel` が高解像度のサブピクセル画像を、Python ループではなく1回の疎行列・ベクトル積でより粗いピクセルグリッドへダウンサンプルする仕組みです。`image_mask` は円形／楕円形の screen の外側にあるピクセル／サブピクセルにマークを付けます（矩形の screen の場合はマスクなし）。これにより表示画像でゼロにできます。

### cosine による減衰と etendue 重み

`Screen.cosine(eye)` は、各**サブピクセル**について、eye の光軸とそのサブピクセルへの直線とのなす角の余弦を計算します。`tangent = |subpixel_position − eye.position| / focal_length` とすると `cosine = 1 / sqrt(1 + tangent²)` です。`etendue_per_subpixel` は小開口近似の診断値 `A_subpixel · cos⁴(θ) / (4π)` として残していますが、有限EyeではsourceとEye内部位置の両方に依存するため、実際のラスタライザはこのdetector側だけの値を再利用しません。

### `ray2image_grid`：光線バンドルを疎な画像へ変換する

`Screen.ray2image_grid` は現在のコードで唯一のラスタライザです（本ドキュメントの以前のバージョンでは `ray2image`／`ray2image2` という別のバリアントにも言及していましたが、それらは削除されており、`Camera.calc_image_vec` にコメントアウトされた呼び出し箇所だけが残っています）。ある `Rays` バンドルに対して、列 `r` が光線 `r` の etendue 重み付きサブピクセル足跡を保持する `(N_subpixel, n_rays)` の疎行列を構築します。アルゴリズムは以下の通りです。

1. **画像座標へ変換する。** `uv = xy2uv(rays.XY)`。
2. **各光線の screen 上でのスポットサイズを計算する。** 各 eye は物理的な `eye_size`（自身の aperture の広がり）を持ちます。ある光線の拡大率のもとでは、**投影された**足跡の半径（または半径・半辺）は `half = 0.5 · eye_size · zoom_rate` です。これは、有限サイズの pinhole／レンズがその奥行きにある点光源に対して screen 上に落とす幾何光学的なぼやけの円板（または楕円・矩形）です——光源点が焦点面から遠いほど `zoom_rate` は大きくなり、ぼやけたスポットも大きくなります。
3. **足跡が screen に一切かからない光線を安価に棄却する。** サブピクセルグリッドの範囲（`u_min..u_max`、`v_min..v_max`）に対する軸並行境界ボックス（AABB）テストによって行います——これは（円形とは限らない）足跡を囲むバウンディングボックスを示す ASCII 図とともにコード中にインラインでドキュメント化されています。
4. **生き残った各光線について、候補サブピクセルの小さなタイルにクリップする**（`i_min..i_max`、`j_min..j_max` は足跡の AABB が重なるサブピクセルインデックスの範囲）。各cellについて、完全内ならcell全面積、完全外なら0、境界なら楕円と矩形または矩形同士の解析的な交差面積を計算します。中心点の内外だけで判定しないため、spotが1 pixelより小さくても`subpixel_resolution=1`で面積を保存できます。screen 全体ではなくAABBタイルだけを調べることが、数百万本の光線に対してスケールする鍵です。
5. **Eye内部位置ごとの局所etendueを積分して疎行列を組み立てる。** spot内のdetector位置 `q` を `a = (q - q_center) / zoom_rate` によりEye内部位置へ逆写像します。Eye中心から見たsourceの横方向offsetを `rho` とすると、局所距離は `D² = Z² + |rho-a|²`、detector面積あたりの重みは `Z / (4π · zoom_rate² · D³)` です。この密度を正確なspot/cell交差領域上で有界な決定論的quadratureにより積分し、その値をCSR/CSCへ格納します。spot全体が大きな1 pixelに入る場合もEye形状上で直接積分するため、局所etendueはdetectorのsubpixel分割に依存しません。

`Screen` のその他のヘルパーは、より単純な座標変換／集約用のユーティリティです：`xy2uv`（カメラの `(X,Y)` → 画像の `(u,v)`。上述）、`uv2subpixel_index`（画像座標 → 整数のサブピクセルインデックス。範囲外のヒットは除外）、`subpixel_to_pixel`（疎なダウンサンプリング。上述）、`show_image`（Matplotlib によるピクセル／サブピクセル画像の表示）。

## Camera：eye・aperture・screen を結びつける

`Camera` は（すべて同じ `eye_type` を共有する）1つ以上の `Eye` インスタンス、`Aperture` オブジェクトのリスト、1つの `Screen` をまとめ、`camera_position` と `rotation_matrix` によってアセンブリ全体をワールド空間に配置・向き付けします。Eye spotはpixel/subpixelより小さくても構いません。解析的な重なり面積で光量を保存するため、Camera生成時にdetector解像度の制約は課しません。

1つのscreenと1つのpinholeからなる一般的な構成には、`Camera.single_pinhole(...)` を利用できます。このファクトリは、screen中心とeye中心をカメラ原点に置き、`camera_position=(0, 0, 0)`、単位回転行列のローカル基準姿勢で光学系を生成します。その後、Camera全体を配置します。

```python
camera = Camera.single_pinhole(
    focal_length=25,
    eye_size=1,
    screen_size=61 * 0.13,
    pixel_shape=(61, 61),
    subpixel_resolution=5,
    apertures=aperture,
).set_rotation_euler(
    "zxz", (2.9, 98, -19), degrees=True,
).set_camera_position(
    world_position,
).translate_camera(
    (4.15, 0, 0),
)
```

`set_camera_position` と `set_rotation_euler` は絶対値を設定します。`translate_world` はワールド座標の相対移動、`translate_camera` は現在の回転を使ってカメラ座標の移動量をワールド座標へ変換する相対移動です。したがって、回転とカメラ座標での相対移動は順序に依存します。CADなどからworld→camera行列を直接得られる場合は `set_rotation_matrix(matrix)` を使えます。最終的なscreen軸が得られる場合は、`set_orientation(look=normal, right=screen_right)`（または `down=`）によって、ワールド座標で与えた軸を直交化して同じ行列を直接構成できます。CADが方向ベクトルではなく座標点を与える場合は、先にCamera位置を設定してから `set_orientation_from_points(look_point=..., right_point=...)`（または `down_point=`）を呼ぶと、各点から `camera_position` を引いて同じ姿勢計算を行います。

Cameraを `World` に登録すると、その光学構成はfreezeされます。Cameraの姿勢と、すべての `Eye`、`Screen`、`Aperture` のジオメトリは変更不可になり、公開されるNumPy配列、screenの疎行列マッピング、STLデータバッファもread-onlyになります。変更メソッドは `RuntimeError` を送出し、`eyes` と `apertures` は外部からコレクションを変更できないtupleとして公開されます。登録済みの構成を変更する場合は、新しいCameraを作って `World.change_camera` で交換します。freezeされたCameraは複数Worldで安全に共有でき、Worldから削除してもfreezeは解除されません。

### `calc_image_vec`：ワールド座標の点から疎な screen 画像へ、ステップごとの解説

`Camera.calc_image_vec(eye_num, points, ...)` は、`World` が（カメラの eye ごとに1回）点のバッチを1つの eye を通して投影するために呼び出す、トップレベルのエントリポイントです。次の3ステップを実行します。

1. **ワールド → カメラ。** `points_in_camera = self.world2camera(points)`。
2. **aperture の可視性判定（省略可能。デフォルトで有効）。** カメラ上のすべての `Aperture` について `stl_utils.check_visible(mesh_obj=aperture.stl_model, start=eye.position, grid_points=points_in_camera, behind_start_included=True)` を実行します。点が `visible` になるのは**すべて**の aperture をクリアした場合だけです（`np.all(visible_list, axis=0)`）——「クリアする」ことの幾何学的な意味は上記の Aperture の節を参照してください。
3. **光線追跡とラスタライズ。** `eye.calc_rays(points_in_camera, visible)` が `Rays` バンドル（上述の pinhole 投影）を生成し、それを `screen.ray2image_grid(eye, rays)` が最終的な `(N_subpixel, n_points)` の疎行列へ変換します。

この行列は、*各点の光線がどのサブピクセルに到達し、どんな etendue 重みを持つか*だけをエンコードしていることに注意してください——各点の発光強度は知りません。これに点ごと（積分後はボクセルごと）の強度ベクトルを掛け合わせることで、初めて実際の画像が得られます。その積分ステップは `multi_pinhole.world.World` が担当します（`docs/world.md` を参照）。

### 具体例：1本の光線を最初から最後まで追跡する

`position=(5, 0)`、`focal_length=20` で作った pinhole eye（`eye.position = (5, 0, 20)`）を、`camera_position=(0, 0, -60)`、単位回転行列のカメラに搭載した場合：

1. ワールド座標の点 `(2, 0, -30)` は、カメラ座標では `(2, 0, -30) − (0, 0, -60) = (2, 0, 30)` になります（単位回転なので変化しません）。
2. eye 座標系では、eye の位置 `(5, 0, 20)` を引いて `(X', Y', Z') = (-3, 0, 10)` になります。`Z' = 10 > 0` なので、この点は eye の前方にあり前方判定を通過します（`Z' = 0` の点はちょうど pinhole 自身の奥行きに位置することになり、投影式がそこでゼロ除算になるため除外されます）。
3. 投影の式から `XY = −(-3, 0)/10 · 20 + (5, 0) = (6 + 5, 0) = (11, 0)`（カメラ座標系）、`zoom_rate = 1 + 20/10 = 3` が得られます。
4. `Screen.xy2uv` は `(11, 0)` を、軸の順序を反転しシフトすることで画像座標へ変換します：`uv = (0, 11) + screen_size/2`。
5. `ray2image_grid` は、この1本の光線を、その `uv` 点から `eye_size/2 · zoom_rate`（ステップ3の拡大率でスケーリングされた、円形 eye の場合は円板）以内にあるすべてのサブピクセルに広げ、上述の etendue 係数で各ヒットに重み付けします。

### 可視化ヘルパー

`draw_optical_system`、`draw_camera_orientation_plotly`、`draw_camera_orientation` は、eye・aperture・screen を Matplotlib または Plotly の3Dシーンに描画し、アライメントやデバッグに役立てます。これらは純粋な可視化であり、上記の投影計算には影響しません。

これらのクラスが、`multi_pinhole.world` と `multi_pinhole.voxel` が完全なマルチ aperture 撮像実験をシミュレートするための土台になっています——`World` が個々の点に対する `calc_image_vec` の呼び出しから、完全なボクセル→スクリーンの投影行列をどう組み立てるかについては `docs/world.md` を参照してください。
