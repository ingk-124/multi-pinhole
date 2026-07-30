# Worldの投影モデル

> **Level 2 — 科学モデル。** 通常の計算は[概要](overview.md)に従えば実行
> できます。このページはsource resolutionの選択、visibilityとprojectionの
> 解釈、数値近似の監査に使います。**内部実装**と明記した部分は読み飛ばせます。

`multi_pinhole.world` モジュールはボクセル、カメラ、そして必要に応じて遮蔽物（STL の「壁」）をまとめ、シミュレートされたシーンを構成します。このモジュールが存在する目的は本質的に2つの計算に集約されます。**各カメラの各 eye からどのボクセルが見えるか（可視性）**と、**ボクセルの発光強度を検出器ピクセル強度へ写像する疎行列（投影行列）**です。本ドキュメントでは、この2つの計算——可視性判定と投影行列の組み立て——を、`multi_pinhole/world.py` の実装に基づいて順を追って説明し、最後にパイプライン全体の具体例を示します。

## 読み方

| 目的 | 読む節 |
|---|---|
| cacheと保存の関係 | scene lifecycle |
| visibilityの0/1/2を解釈 | visibility model |
| `res`, `partial_res`, adaptiveを選択 | 投影精度とAPI |
| `P`, `P.T`, array shapeを理解 | 投影精度とAPI |
| chunking・sparse assemblyを監査 | 内部実装 |

## scene lifecycle

### Worldの構築

`World.__init__` はオプションのボクセル、カメラ、壁、`inside_func` 引数を受け取ります。入力が省略された場合は既定値（空の `Voxel()`、カメラなし、壁なし）にフォールバックし、`voxel.set_world(self)`・`camera.set_world(self)` によって直ちにワールドへ再接続されるため、可視性の判定結果など共有状態を各コンポーネントが参照できるようになります。カメラはstable keyのマッピングへ正規化されます。listを渡すと `range(len(cameras))` の整数keyが割り当てられ、dictを渡すと `{"left": camera_left, "right": camera_right}` のような明示keyが維持されます。Cameraを削除しても残りのkeyは再採番されず、`add_camera(key, camera)` ではkeyの指定が必須です。`world.cameras` はread-only mappingとして公開され、変更は `add_camera`、`change_camera`、`remove_camera` を通して行います。明示的なkeyリセット機能はfuture workとしています。カメラごとの可視フラグ（`_visible_vertices`、`_visible_voxels`）と投影行列（各 eye ごとの `_projection`、カメラ全体で集約した `_P_matrix`）にも同じkeyを使う並行ディクショナリが確保されます——いずれも対応する計算が実行されるまでは `None` のままです。`inside_func` を与えると `set_inside_vertices` が即座に呼ばれ内部頂点マスクが初期化されます。指定しない場合は「すべての頂点が内部」という遅延初期化のままです（後述の `inside_vertices` を参照）。

壁は `stl.mesh.Mesh` オブジェクトのリストに正規化されます。変更（`walls` セッター）があるとキャッシュを無効化し、`update_min` と `update_max` を通じて事前計算済みのメッシュ境界を更新し、後のプロットに備えて結合した軸方向の限界値（`wall_ranges`）を保存します。

### sceneの確認と永続化

`camera_info` と `voxel_info` は登録済みセンサーとグリッドを要約します。
scene構築と計算済みcheckpointには、独立した形式を使います。

- `World.from_config(path_or_mapping)` と `world.to_config(path)` はcacheを
  含まないJSON scene schemaを使います。詳細は
  [config schema reference](config.md)を参照してください。
- `world.save(path)`、`World.inspect_archive(path)`、`World.load(path)` は、
  平文JSON manifestとdill payloadからなるversion付きarchiveを使います。
  `save_world`と`load_world`は互換aliasとして残します。詳細は
  [serialization reference](serialization.md)を参照してください。

archive manifestではlibrary version、World serialization schema、projection
cache schemaを分離します。schema 3のprojection cacheは再利用します。非互換な
projection cache schemaを読み込んだ場合、再利用可能なvisibilityを維持しながら
`_projection`と`_P_matrix`だけを無効化します。旧direct-dill Worldは読み込んだ直後に
新archiveへ保存できます。pickle/dillは任意codeを実行できるため、信頼できるfileだけを
読み込んでください。

camera、voxel、wallのproperty setterは可能な限りcache済みのvisibility／projectionを
再利用します。不可能な場合は`_invalidate_visibility_cache()`を呼び、
`_visible_vertices`、`_visible_voxels`、`_projection`、`_P_matrix`を`None`へ戻します。

## visibility model

通常利用では`find_visible_voxels(camera_idx)`を呼びます。返り値は
`(N_eye, N_voxel)`で、`0=不可視`, `1=部分可視`, `2=完全可視`です。
分類はvoxelの8頂点に基づき、部分可視voxelは投影時に内部を再sampleします。

これは解析的な可視体積計算ではありません。境界は最終的にsubvoxel中心で
近似されるため、wall、aperture、`inside`境界が結果に効く場合はresolution
sweepが必要です。

> **内部実装:** この節の残りはpoint→vertex→voxel maskの作り方です。数値設定
> だけを決める場合は[投影精度とAPI](#投影精度とapi)へ進んで構いません。

`World` はscene状態とcameraごとのvisibility cacheを所有し、privateな
`multi_pinhole._visibility` moduleはgeometryからmaskを求める計算だけを
担当します。公開entry pointは引き続き `World.find_visible_points` です。
ここでcamera keyを解決し、pointとwallをcamera座標へ準備します。
`_visible_vertices` と `_visible_voxels` への代入、およびgeometry変更時の
projection cache無効化は、引き続き `World` だけが担当します。

可視性は **点 → 頂点 → ボクセル** という3段階の粒度で計算され、それぞれが前段の結果の上に構築されます。

### `find_visible_points`：eye ごとの可視性判定の中核

`find_visible_points(points, camera_idx, eye_idx=None)` は、それ以外のすべての可視性計算が呼び出す基本ルーチンです。指定したカメラおよびそのカメラが持つ各 eye について、以下を行います。

1. `points`（ワールド座標）を `camera.world2camera` でそのカメラの座標系に変換し、各壁メッシュも同じ座標系へコピーする（`stl_utils.copy_model(wall, -camera_position, rotation.T)`）。これにより、以降の判定はすべて一貫した1つの座標系上で行われます。
2. 光軸方向で eye より手前にある点だけを暫定的に可視とマークします：`camera_points[:, 2] >= eye.position[-1]`。
3. カメラ上のすべての `Aperture` について `stl_utils.check_visible(mesh_obj=aperture.stl_model, start=eye.position, grid_points=camera_points, behind_start_included=True)` を実行し、**すべての** aperture をクリアした点だけを可視とします（`np.all(..., axis=0)`）——aperture はモデル化された開口部を通らない限り光線を遮る不透明な面として扱われます。これは `Camera.calc_image_vec` が aperture を扱う方法（`docs/core.md` 参照）と厳密に一致しており、ボクセル単位の可視性判定と光線単位のレンダリングの整合性を保つ要となっています。
4. すべての壁メッシュについて同様の `check_visible` テスト（今回は `behind_start_included` なし。壁は aperture 平面のような特殊な扱いではなく、通常の不透明な形状であるため）を実行し、結果を AND で合成します。

結果は `(N_eye, N_points)` の真偽値行列です。`check_visible` 自体は `multi_pinhole.utils.stl_utils` に実装された2段階の幾何学的テスト（コーンによる事前フィルタ、その後の厳密な Möller–Trumbore 三角形交差判定）であり、eye からある点までの線分がメッシュを横切るかどうかをどのように判定しているかは `docs/utilities.md` を参照してください。

### 点から頂点へ、頂点からボクセルへ

すべてのボクセルの内部を直接テストするのはコストが高いため、ワールドはボクセルグリッドの**頂点**を一度だけテストし、その結果を頂点を共有するすべてのボクセルで再利用します。

* `_find_visible_vertices` はグリッド頂点（`self.voxel.grid`）に対して `find_visible_points` を呼び出しますが、`inside_vertices` で `True` とフラグが立っている頂点のみを対象とします——モデル化された体積の外側にある頂点は、一度も光線追跡されることなく `False` のままになります。結果はカメラごとに `_visible_vertices` として `(N_eye, N_grid_vertices)` の真偽値配列にキャッシュされます。
* `find_visible_voxels` はこの頂点単位の結果を各ボクセルの8個の角（`self.voxel.vertices_indices`）に集約し、`(eye, voxel)` の組ごとに次の3状態のいずれかを報告します。

  * **`0` — 不可視**：ボクセルの8個の角頂点のうち可視なものが一つもない。
  * **`1` — 部分的に可視**：一部の角は可視だが全部ではない（そのボクセルは aperture のエッジや壁のシルエットなど、遮蔽境界をまたいでいる）。
  * **`2` — 完全に可視**：8個の角すべてが可視。以降の投影パイプラインはこのボクセルの内部を再テストせず、直接積分に進むことができます。

`set_inside_vertices(function)` は、そもそも「モデル化された体積」をどう定義するかを指定する手段です。`function` はボクセルグリッドの `(x, y, z)` 座標に対して評価され、グリッド頂点上の真偽値マスク（例えば「トーラス内部」「真空容器内部」）を返す必要があります。このマスクの外側にある頂点は可視性・投影計算から完全に除外されます。これは正確性のためのツールであると同時に（物理デバイスの外側からの発光をレンダリングしないため）、大部分が空の空間であるグリッドに対しては大きな性能最適化にもなります。

## 投影精度とAPI

`set_projection_matrix(res, ...)` は、`Voxel` グリッドと可視ボクセルの情報を、すべてのカメラ・すべての eye についてボクセル強度を検出器信号へ写像する疎行列に変換するエントリポイントです。各 `(camera, eye)` の組について `_calc_voxel_image_for_eye` を呼び出し、その後1つのカメラ上のすべての eye をそのカメラのピクセル空間 `P_matrix` へ集約します。

source体積積分はsubvoxel中心を使う複合midpoint quadratureです。emissionは
voxel center値から三線形補間され、各sampleはowner voxelの体積/sample数で
重み付けされます。定数fieldを保存し、内部cellではaffine fieldを再現しますが、
外端half-cellはnearest centerにclampされます。partial visibilityと`inside`
境界はsample centerのBoolean判定で、解析的な切断体積積分ではありません。

adaptive resolutionはlocal perspective scaleに基づくgeometry heuristicであり、
画像誤差の上限ではありません。`point_source_threshold`は誤差許容値ではなく、
`partial_res`も境界精度を保証しません。最終結果はresolution sweepで収束確認
してください。

重い計算を開始する前に、同じsource resolution設定で `preflight_projection` を実行できます。

```python
work = world.preflight_projection(
    res=5,
    res_mode="auto",
    partial_res=3,
)
print(work.summary())
print(work.total_samples_upper_bound)
```

reportはeyeごとに完全可視・部分可視voxel数を分け、完全可視voxelの採用res bucket、adaptive時のideal res分位点と上限clipされた軸数を整理します。総sample数は、完全可視側については正確な値、部分可視側についてはpoint visibilityとinside maskを適用する前の保守的な上限です。実行時間や疎行列`nnz`の予測値ではありません。preflightはvoxel visibilityを計算・cacheしますが、`projection`や`P_matrix`は構築・変更しません。後続の実計算はvisibility cacheを再利用できます。

構築後は`world.project(emission, camera_idx, eye_idx=None)`でcamera合算行列、
または1つのEye行列を適用します。`world.backproject(...)`は同じ行列の転置を
適用する離散随伴で、逆問題の解や逆行列ではありません。vectorと列方向batch
`(N_voxel, N_rhs)` / `(N_pixel, N_rhs)`を扱えます。対象行列がcacheされて
いなければ、暗黙に構築せず`RuntimeError`を送出します。

> **通常の科学利用はここまでで十分です。** 以下は同じ契約の実装と最適化です。

## 内部実装

projection設定とcache lifecycleは`World`が所有します。privateな
`multi_pinhole._projection_matrix`は明示的なgeometry入力からoptical-bin
quadratureとsparse assemblyを行い、World cacheを変更せずCSRを返します。
contiguous-voxel経路はadaptive scheduling、visibility callback、parallel taskを
まとめて調整するため、引き続き`World`にあります。

### `_calc_voxel_image_for_eye`：完全可視ボクセルと部分可視ボクセル

このモジュール内で最もコストが高く、かつ最も中核的な計算です。前段で計算したボクセルの可視性に基づき、ボクセルを2つのグループに分けて異なる方法で処理します。完全可視ボクセルはこれ以上の光線追跡を必要としないためです。

* **完全可視ボクセル（`vis_flag == 2`）**：ボクセルごとに `res` 個のサブボクセル点をサンプリングし（`Voxel.get_sub_voxel_centers`）、そのすべてを `Camera.calc_image_vec(..., check_visibility=False)` で eye を通して投影します（可視性は既知なので、コストの高い aperture／壁の遮蔽判定はスキップされます）。得られたサブボクセル画像を、後述の補間行列 `S` と組み合わせて、ボクセルごとに1列を生成します。
* **部分可視ボクセル（`vis_flag == 1`）**：同じサブボクセル点をサンプリングしますが、まずそのサブボクセル中心点に対して改めて `find_visible_points` を実行します（親ボクセルの8つの角がすべて一致していなくても、内部の一部が遮蔽されている可能性があるため）。不可視なサンプルをマスクで除外し、生き残ったサンプルのみを投影します。

どちらの経路も `_sub_voxel_interpolator_matrix` を通り、**ボクセル中心の値**を重み付きサブボクセルサンプルへ直接写像する行列 `S` を構築します。内部の各サンプルは周囲の最大8個のボクセル中心から三線形補間され、格子外周の半ボクセル領域では最寄りの中心値へclampされます。さらに各行を `voxel.volume / samples_per_voxel` でスケーリングします。このスケーリングによって、サブボクセル行にわたる和がボクセル体積にわたる投影信号の**積分**の近似になります。定数profileの総量を保ち、格子内部では一次profileを再現しながら、以前の「中心→頂点→サブボクセル」方式より局所的な補間になります。

具体的には、あるボクセルのバッチについて、永続化するeyeごとのpixel投影は

```
P_eye = T_pixel_from_subpixel @ calc_image_vec(eye, sub_voxel_centers) @ S
```

と表せます。ここで `calc_image_vec`（`docs/core.md` 参照）は `(N_subpixel, N_sub_voxel_samples)` の光線追跡・ラスタライズ行列、`T` は厳密な `(N_pixel, N_subpixel)` のdetector binning行列、`S` は上記の `(N_sub_voxel_samples, N_voxel_batch)` の補間・積分行列です。したがって `P_eye` は `(N_pixel, N_voxel_batch)` になります。subpixelでの面積積分精度は保ちますが、subpixel行そのものは永続化しません。

### チャンク分割と並列化

すべてのボクセルのサブボクセルサンプルに対して一度に `calc_image_vec` を実体化するとメモリを圧迫しかねません（1本の光線が多数のサブピクセルに触れうるため）。これを抑えるために、この関数は以下を行います。

1. **疎度の推定**：少数（20個、それより少なければその数）のボクセルをランダムサンプリングして `calc_image_vec` を実行し、ボクセルあたりの非ゼロ要素数（`nnz`）の平均を測定します。
2. **バッチサイズの決定**：サンプル点、画像行列、補間行列、結果行列の実バイト数を小規模サンプルから推定し、同時実行タスク全体が `max_working_memory`（デフォルト10億byte）に収まるように選びます。従来の `max_nnz` も第二の上限として残します。
3. **チャンクを直列またはスレッドプールで処理**：`n_jobs > 1` の場合は `ThreadPoolExecutor` を使い、同時に保持するタスクを最大 `2 * n_jobs` に制限します。各ワーカー内でサンプル点と補間行列を生成し、完了したfutureを直ちに回収します。COO形式の結果bufferは固定チャンク数ではなく、`max_working_memory` から決めたbyte上限へ達した時点で畳み込みます。これにより、すべてのチャンクの入力と出力を同時に保持せず、ピークメモリを抑えます。

この一連の処理（手順1〜3。完全可視・部分可視の各グループに対して別々に実行されます）は純粋にメモリ／スループットのトレードオフのために存在しています——数学的な結果は `n_jobs` や `max_nnz` に関わらず同じ疎行列になります。変わるのは計算の分割方法だけで、答えではありません。

`res`は必須引数です。`res_mode="fixed"`では指定値を固定resとして使い、`res_mode="auto"`では完全可視voxelごとのideal resに対する軸別上限として使います。voxelの外接球をoff-axisの `1/cos(theta)` を含む局所worst-case倍率でscreenへ投影し、detector subpixel pitchと局所有限Eye PSF幅で無次元化します。`point_source_threshold`のdefaultは `1/8`です。これは幾何学heuristicであり画像誤差の上限ではありません。上限なし計算は `res=None, res_mode="ideal"` と固定`partial_res`を明示した場合だけ許可します。部分可視voxelはvisibilityが不連続なので固定`partial_res`を使い、`fixed`または`auto`で省略した場合は`res`を再利用します。ただし任意位置のwall/inside境界に対して、小さな固定`partial_res`は画像誤差を保証しません。対象geometryで別途収束確認するか、保守的な値を明示します。

各subvoxelの投影像には、chunkの組み立て中にスクリーンの `transform_matrix`（subpixel→pixelへのビニング。`docs/core.md` 参照）を直ちに適用します。eyeごとのpixel空間の結果を `self._projection[camera_idx][eye_idx]` に格納し、`set_projection_matrix` はすべてのeyeを合算して `self._P_matrix[camera_idx]` を生成します。subpixel行は積分中だけの一時データであり、projection cacheには保持しません。

### `trace_line`：完全な行列を構築せずに少数の点を投影する

「この特定の点はスクリーン上のどこに写るか」といった簡単な確認を、投影パイプライン全体を実行せずに行いたい場合、`trace_line(points, camera_idx, eye_idx, coord_type)` は `points` を1つの eye を通して投影し、カメラ平面の `XY` 座標かスクリーンの `UV` ピクセル座標のいずれかを返します。`calc_image_vec` と異なり、aperture／壁の可視性判定やサブピクセルへのラスタライズは行いません——`Eye.calc_rays` の薄いラッパーであり、レンダリングのためではなく幾何のデバッグのために有用です。

## 関連する作業ガイド

実行workflowは重複させず[概要と最初の投影](overview.md)に集約しています。
camera姿勢、voxel field、検出器画像の描画は[可視化](visualization.md)を
参照してください。
