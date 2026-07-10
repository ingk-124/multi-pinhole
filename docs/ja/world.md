# ワールドモジュールガイド

`multi_pinhole.world` モジュールはボクセル、カメラ、そして必要に応じて遮蔽物（STL の「壁」）をまとめ、シミュレートされたシーンを構成します。このモジュールが存在する目的は本質的に2つの計算に集約されます。**各カメラの各 eye からどのボクセルが見えるか（可視性）**と、**ボクセルの発光強度を検出器ピクセル強度へ写像する疎行列（投影行列）**です。本ドキュメントでは、この2つの計算——可視性判定と投影行列の組み立て——を、`multi_pinhole/world.py` の実装に基づいて順を追って説明し、最後にパイプライン全体の具体例を示します。

## ヘルパーユーティリティ

配列の管理を円滑にするヘルパーがあります。

* `multi_pinhole.utils.type_check_and_list`（このモジュールでは定義されておらず `multi_pinhole.utils` からインポートされる）は単一オブジェクトまたはリストを受け取り、要素型を強制しながらリストに変換し、`None` の場合のデフォルト値も指定できます。【F:multi_pinhole/utils/__init__.py†L24-L54】`cameras`・`walls` など複数のセッターを実際に支えており、呼び出し側が単体または複数のインスタンスを渡せるようにします。
* このモジュールにローカルで定義されている `type_list` は同様の正規化を行いますが、現在の `world.py` 内ではどこからも呼び出されておらず、`type_check_and_list` が切り出される前の名残として事実上未使用（デッドコード）です。【F:multi_pinhole/world.py†L36-L74】
* `_blocks_lengths` と `_slice_blocks` は点群および疎行列ブロックのコレクションを扱い、後続の投影処理のために軽量なスライシング機能を提供します。【F:multi_pinhole/world.py†L77-L148】

## ワールドの構築

`World.__init__` はオプションのボクセル、カメラ、壁、`inside_func` 引数を受け取ります。【F:multi_pinhole/world.py†L162-L228】入力が省略された場合は既定値（空の `Voxel()`、カメラなし、壁なし）にフォールバックし、`voxel.set_world(self)`・`camera.set_world(self)` によって直ちにワールドへ再接続されるため、可視性の判定結果など共有状態を各コンポーネントが参照できるようになります。カメラはインデックスマッピング（`{int: Camera}`、`self._cameras`）に正規化され、カメラごとの可視フラグ（`_visible_vertices`、`_visible_voxels`）と投影行列（各 eye ごとの `_projection`、カメラ全体で集約した `_P_matrix`）をキャッシュする並行ディクショナリが確保されます——いずれも対応する計算が実行されるまでは `None` のままです。`inside_func` を与えると `set_inside_vertices` が即座に呼ばれ内部頂点マスクが初期化されます。指定しない場合は「すべての頂点が内部」という遅延初期化のままです（後述の `inside_vertices` を参照）。

壁は `stl.mesh.Mesh` オブジェクトのリストに正規化されます。変更（`walls` セッター）があるとキャッシュを無効化し、`update_min` と `update_max` を通じて事前計算済みのメッシュ境界を更新し、後のプロットに備えて結合した軸方向の限界値（`wall_ranges`）を保存します。

## シーンの内省と永続化

`camera_info` と `voxel_info` は登録済みのセンサーおよびグリッドの概要を提供します。`save_world`/`load_world` は `dill`（`coordinate_transform` が内部で使うクロージャなども含めてシリアライズできる、`pickle` 互換のライブラリ）を用いてシーン全体をシリアライズし、長時間のシミュレーションを容易にチェックポイントできます。【F:multi_pinhole/world.py†L283-L311】カメラ、ボクセル、壁のプロパティセッターは可能な限りキャッシュ済みの可視性・投影データを再利用しますが、それが不可能な場合は `_invalidate_visibility_cache()` を呼び出し、`_visible_vertices`・`_visible_voxels`・`_projection`・`_P_matrix` を `None` のプレースホルダーへリセットして、次回のクエリで最初から再計算させます。【F:multi_pinhole/world.py†L376-L387】

## 可視性の評価

可視性は **点 → 頂点 → ボクセル** という3段階の粒度で計算され、それぞれが前段の結果の上に構築されます。

### `find_visible_points`：eye ごとの可視性判定の中核

`find_visible_points(points, camera_idx, eye_idx=None)` は、それ以外のすべての可視性計算が呼び出す基本ルーチンです。【F:multi_pinhole/world.py†L639-L716】指定したカメラおよびそのカメラが持つ各 eye について、以下を行います。

1. `points`（ワールド座標）を `camera.world2camera` でそのカメラの座標系に変換し、各壁メッシュも同じ座標系へコピーする（`stl_utils.copy_model(wall, -camera_position, rotation.T)`）。これにより、以降の判定はすべて一貫した1つの座標系上で行われます。
2. 光軸方向で eye より手前にある点だけを暫定的に可視とマークします：`camera_points[:, 2] >= eye.position[-1]`。
3. カメラ上のすべての `Aperture` について `stl_utils.check_visible(mesh_obj=aperture.stl_model, start=eye.position, grid_points=camera_points, behind_start_included=True)` を実行し、**すべての** aperture をクリアした点だけを可視とします（`np.all(..., axis=0)`）——aperture はモデル化された開口部を通らない限り光線を遮る不透明な面として扱われます。これは `Camera.calc_image_vec` が aperture を扱う方法（`docs/core.md` 参照）と厳密に一致しており、ボクセル単位の可視性判定と光線単位のレンダリングの整合性を保つ要となっています。
4. すべての壁メッシュについて同様の `check_visible` テスト（今回は `behind_start_included` なし。壁は aperture 平面のような特殊な扱いではなく、通常の不透明な形状であるため）を実行し、結果を AND で合成します。

結果は `(N_eye, N_points)` の真偽値行列です。`check_visible` 自体は `multi_pinhole.utils.stl_utils` に実装された2段階の幾何学的テスト（コーンによる事前フィルタ、その後の厳密な Möller–Trumbore 三角形交差判定）であり、eye からある点までの線分がメッシュを横切るかどうかをどのように判定しているかは `docs/utilities.md` を参照してください。

### 点から頂点へ、頂点からボクセルへ

すべてのボクセルの内部を直接テストするのはコストが高いため、ワールドはボクセルグリッドの**頂点**を一度だけテストし、その結果を頂点を共有するすべてのボクセルで再利用します。

* `_find_visible_vertices` はグリッド頂点（`self.voxel.grid`）に対して `find_visible_points` を呼び出しますが、`inside_vertices` で `True` とフラグが立っている頂点のみを対象とします——モデル化された体積の外側にある頂点は、一度も光線追跡されることなく `False` のままになります。結果はカメラごとに `_visible_vertices` として `(N_eye, N_grid_vertices)` の真偽値配列にキャッシュされます。【F:multi_pinhole/world.py†L718-L770】
* `find_visible_voxels` はこの頂点単位の結果を各ボクセルの8個の角（`self.voxel.vertices_indices`）に集約し、`(eye, voxel)` の組ごとに次の3状態のいずれかを報告します。【F:multi_pinhole/world.py†L772-L804】

  * **`0` — 不可視**：ボクセルの8個の角頂点のうち可視なものが一つもない。
  * **`1` — 部分的に可視**：一部の角は可視だが全部ではない（そのボクセルは aperture のエッジや壁のシルエットなど、遮蔽境界をまたいでいる）。
  * **`2` — 完全に可視**：8個の角すべてが可視。以降の投影パイプラインはこのボクセルの内部を再テストせず、直接積分に進むことができます。

`set_inside_vertices(function)` は、そもそも「モデル化された体積」をどう定義するかを指定する手段です。`function` はボクセルグリッドの `(x, y, z)` 座標に対して評価され、グリッド頂点上の真偽値マスク（例えば「トーラス内部」「真空容器内部」）を返す必要があります。このマスクの外側にある頂点は可視性・投影計算から完全に除外されます。これは正確性のためのツールであると同時に（物理デバイスの外側からの発光をレンダリングしないため）、大部分が空の空間であるグリッドに対しては大きな性能最適化にもなります。

## 投影の組み立て

`set_projection_matrix(res, ...)` は、`Voxel` グリッドと可視ボクセルの情報を、すべてのカメラ・すべての eye についてボクセル強度を検出器信号へ写像する疎行列に変換するエントリポイントです。【F:multi_pinhole/world.py†L1182-L1239】各 `(camera, eye)` の組について `_calc_voxel_image_for_eye` を呼び出し、その後1つのカメラ上のすべての eye をそのカメラのピクセル空間 `P_matrix` へ集約します。

### `_calc_voxel_image_for_eye`：完全可視ボクセルと部分可視ボクセル

このモジュール内で最もコストが高く、かつ最も中核的な計算です。【F:multi_pinhole/world.py†L806-L1145】前段で計算したボクセルの可視性に基づき、ボクセルを2つのグループに分けて異なる方法で処理します。完全可視ボクセルはこれ以上の光線追跡を必要としないためです。

* **完全可視ボクセル（`vis_flag == 2`）**：ボクセルごとに `res` 個のサブボクセル点をサンプリングし（`Voxel.get_sub_voxel_centers`）、そのすべてを `Camera.calc_image_vec(..., check_visibility=False)` で eye を通して投影します（可視性は既知なので、コストの高い aperture／壁の遮蔽判定はスキップされます）。得られたサブボクセル画像を、後述の補間行列 `S` と組み合わせて、ボクセルごとに1列を生成します。
* **部分可視ボクセル（`vis_flag == 1`）**：同じサブボクセル点をサンプリングしますが、まずそのサブボクセル中心点に対して改めて `find_visible_points` を実行します（親ボクセルの8つの角がすべて一致していなくても、内部の一部が遮蔽されている可能性があるため）。不可視なサンプルをマスクで除外し、生き残ったサンプルのみを投影します。

どちらの経路も `_sub_voxel_interpolator_matrix` を通り、**ボクセルの角の値**を重み付きサブボクセルサンプルへ写像する行列 `S` を構築します。具体的には、各ボクセルの8個の角頂点インデックスを調べ、`Voxel.interpolate_matrix_from_vertices(res)`（`docs/overview.md` の「ボクセルグリッドの幾何」で説明）が与える三線形補間の重みを適用してサブボクセルのサンプル値を合成し、さらに各行を `voxel.volume / samples_per_voxel` でスケーリングします。このスケーリングこそが、あるボクセルのサブボクセル行にわたる和を、そのボクセル体積にわたる投影信号の**積分**の近似に変換している部分です——`res` を増やすと（より多く、より密なサブボクセルサンプルによって）求積が精緻化されるだけで、積分された信号の総量は変化しません。これは `res` を増やしたときに収束してほしい性質そのものです。

具体的には、あるボクセルのバッチについて、eye ごとのサブピクセル画像は

```
I_subpixel = calc_image_vec(eye, sub_voxel_centers)  @  S
```

と表せます。ここで `calc_image_vec`（`docs/core.md` 参照）は `(N_subpixel, N_sub_voxel_samples)` の光線追跡・ラスタライズ行列であり、`S` は上記の `(N_sub_voxel_samples, N_voxel_batch)` の補間・積分行列です。したがって `I_subpixel` は `(N_subpixel, N_voxel_batch)` となり、これがそのボクセルバッチに対する投影行列への eye ごとの寄与そのものになります。

### チャンク分割と並列化

すべてのボクセルのサブボクセルサンプルに対して一度に `calc_image_vec` を実体化するとメモリを圧迫しかねません（1本の光線が多数のサブピクセルに触れうるため）。これを抑えるために、この関数は以下を行います。

1. **疎度の推定**：少数（20個、それより少なければその数）のボクセルをランダムサンプリングして `calc_image_vec` を実行し、ボクセルあたりの非ゼロ要素数（`nnz`）の平均を測定します。
2. **バッチサイズの決定**：`batch_size × est_nnz` が `max_nnz`（デフォルト `100_000_000`）を `n_jobs` 個の並列ワーカーで割った値を下回るように選びます。
3. **チャンクを直列またはスレッドプールで処理**：`n_jobs > 1` の場合は `ThreadPoolExecutor` を使い、チャンクごとに1つのタスクを投入します。各タスクは完全な疎行列オブジェクトではなく COO 形式の `(data, row, col)` タプルを返し、`_process_tasks` が `concurrent.futures.as_completed` を通じて完了したフューチャーを取り出しながら、バッファしたタプルを（10チャンクごとに）定期的に `scipy.sparse.coo_matrix` の和へ畳み込んでいきます。これにより、すべてのチャンクの結果を同時に保持するのではなく、ピークメモリを抑えられます。

この一連の処理（手順1〜3。完全可視・部分可視の各グループに対して別々に実行されます）は純粋にメモリ／スループットのトレードオフのために存在しています——数学的な結果は `n_jobs` や `max_nnz` に関わらず同じ疎行列になります。変わるのは計算の分割方法だけで、答えではありません。

eye ごとの結果は `self._projection[camera_idx][eye_idx]` に格納されます。`set_projection_matrix` はその後、1つのカメラ上のすべての eye を合算し、スクリーンの `transform_matrix`（サブピクセル→ピクセルへのビニング。`docs/core.md` 参照）を適用して、最終的なピクセル空間の行列を `self._P_matrix[camera_idx]` に生成します。【F:multi_pinhole/world.py†L1228-L1234】

### `trace_line`：完全な行列を構築せずに少数の点を投影する

「この特定の点はスクリーン上のどこに写るか」といった簡単な確認を、投影パイプライン全体を実行せずに行いたい場合、`trace_line(points, camera_idx, eye_idx, coord_type)` は `points` を1つの eye を通して投影し、カメラ平面の `XY` 座標かスクリーンの `UV` ピクセル座標のいずれかを返します。【F:multi_pinhole/world.py†L1147-L1180】`calc_image_vec` と異なり、aperture／壁の可視性判定やサブピクセルへのラスタライズは行いません——`Eye.calc_rays` の薄いラッパーであり、レンダリングのためではなく幾何のデバッグのために有用です。

## 可視化

`draw_camera_orientation` は、ボクセル境界、カメラ姿勢（各 `Camera.draw_camera_orientation` に委譲）、登録された壁を1つの3D Matplotlib プロットに重ね合わせます。軸の範囲はデフォルトでボクセルと壁の範囲（10%拡張）を結合したものになりますが、キーワード引数で上書きできます。【F:multi_pinhole/world.py†L1241-L1307】

## 具体例：可視性 → 投影 → 画像

パイプライン全体を実際に流してみます（実行可能な版は `examples/small_voxel_projection.py`、完全なセットアップコードは `docs/overview.md` を参照）。

1. `world = World(voxel=voxel, cameras=[camera])` により、3×3×3 のボクセルグリッドと1台のカメラ（pinhole eye を1つ搭載）を登録します。
2. `world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))` によって、すべてのグリッド頂点を「内部」としてマークします——`inside_func` を渡さなくても同じデフォルトが自動的に適用されますが、ここでは明示的に行っています。
3. `world.set_projection_matrix(res=1, parallel=1)` を呼ぶと、eye ごとに次が実行されます。
   * `_find_visible_vertices` がグリッドの 4×4×4 = 64 個の頂点をカメラの aperture に対して光線追跡します（ここには壁はありません）。結果は `(1, 64)` の真偽値配列としてキャッシュされます。
   * `find_visible_voxels` がその64個の頂点結果を27個のボクセルそれぞれの8つの角に集約し、`(1, 27)` の 0/1/2 可視性フラグの配列を生成します。
   * `_calc_voxel_image_for_eye` がボクセルごとに1個のサブボクセル中心をサンプリングし（`res=1` なので、サブボクセル中心はボクセル自身の重心と一致します）、完全可視ボクセルの中心を `calc_image_vec` で eye を通して投影します（部分可視ボクセルがあれば、投影の前にサンプル単位で可視性を再確認します）。
   * eye ごとのサブピクセル行列は `world.projection[0][0]`（形状 `(N_subpixel, 27)`）に格納されます。`set_projection_matrix` はその後スクリーンのピクセルビニング変換を適用し、`world.P_matrix[0]`（形状 `(N_pixel, 27)`）を生成します。
4. 任意の27要素の `emission` ベクトルに対して、`world.P_matrix[0] @ emission` がシミュレートされたピクセル画像になります——ジオメトリ（カメラ、ボクセルグリッド、aperture、壁、あるいは内部頂点マスク）が変わらない限り、追加の光線追跡は不要です。ジオメトリが変わった場合は対応するキャッシュが無効化され、次の `set_projection_matrix` 呼び出しで変更された部分だけが再計算されます。
