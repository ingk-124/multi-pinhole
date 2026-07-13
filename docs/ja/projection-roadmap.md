# Projection matrix 改善候補

この文書は、projection matrix の高速化と検証作業で残っている候補を記録する。
現在の実装済み範囲と、今後の仕様検討を混同しないための追跡用メモである。

## main 統合前に確認する項目

### Projection cache の互換性

- projection 計算アルゴリズムと cache schema のバージョンを `World` に保存する。
- バージョンが存在しない、または一致しない古い world を読み込んだ場合は、
  `_projection` と `_P_matrix` を無効化して再計算させる。
- 行列 shape の一致だけでは cache の有効性を判断しない。
- 古い pickle に存在しない `Screen` の派生値、特に
  `_subpixel_u_axis` と `_subpixel_v_axis` を読み込み時に再構築する。
- 現行形式、旧形式、不一致バージョンについて回帰テストを追加する。

### MST 実形状での数値検証

まず `d = 75 mm` 程度の短時間で計算できる格子を使い、次の emission
profile を投影する。

1. inside で一定、outside でゼロ
2. 小半径方向に滑らかな Gaussian
3. plasma 境界付近まで値を持つ profile
4. 非対称な profile

各 profile について、総光量、pixel ごとの差、相対 L2 誤差、画像断面を比較する。
detector の `subpixel_resolution = 1, 2, 5` について、総和だけでなく画像分布も
収束していることを確認する。rectangle Eye は、旧実装の spot 幅の不具合修正に
よって結果が変わるため、個別に確認する。

検証後、実際の MST 解析スクリプトで detector 側
`subpixel_resolution = 1` を標準値にできるか決定する。元の約 110 万 voxel、
`d = 25 mm` 相当の計算は、正しさを確認した後に最終 benchmark として一度だけ
実行する候補とする。

### inside 境界と未知変数の扱い

- inside voxel の補間に outside 側の隣接 voxel 値が混ざる場合を調べる。
- outside の emission を常にゼロとするのか、projection matrix の列から除外するのか、
  inverse problem 側で拘束するのかを仕様化する。
- 定数 profile の総光量保存と、境界を横切る profile の挙動をテストする。

## Source subvoxel 解像度

### `res` の自動判定

現在の評価スクリプトは診断用であり、ライブラリに `res="auto"` はない。
自動判定では、代表的な visible voxel に対して `res=1` と `res=2`、必要なら
`res=2` と `res=4` の投影結果を比較し、相対誤差が閾値以下になった最小値を採用する。

- 可能なら軸別に `(r_x, r_y, r_z)` を決める。
- fully visible voxel と partial voxel を別々に評価する。
- 判定に使用した voxel、誤差、採用 res、上限到達の有無を診断情報として返す。
- wall や inside 境界にかかる partial voxel は不連続性を含むため、fully visible
  voxel 用の幾何学的な閾値だけで安全と判断しない。

### voxel ごとの適応 res

遠方かつ fully visible で、voxel の投影像が pinhole PSF より十分小さい場合は
`res=1` とし、partial voxel や投影変化の大きい voxel だけ高い res を使う。
実装する場合は res ごとに voxel を bucket 化し、同じ kernel をまとめて実行する。

## Fine voxel grid を現実的に扱う方法

### 格子幅の比較

emission の空間分解能と有限体積積分の精度を分離するため、少なくとも次を比較する。

- `d=25 mm`, source res 1 または 2
- `d=12.5 mm`, source res 1
- `d=10 mm`, source res 1

`d=10 mm` の bounding grid は約 `405 x 405 x 105`、約 1720 万 voxel になるため、
projection 計算時間だけでなく、可視性配列、行列の列数、メモリ使用量が律速になる。

### Active voxel の compact mapping

torus 外、inside 外、または全 camera から常に不可視な voxel を projection matrix の
列から除外し、compact index と元の voxel index の対応を保存する。inverse problem と
可視化では、この mapping を通して fine grid に戻せるようにする。`d=10 mm` 対応では
優先度が高い。

### Adaptive voxel merging / reduced basis

fine grid 上の emission `f` を reduced parameter `c` と mapping `B` により

```text
f = B c
g = P_fine B c
```

と表す。fully visible かつ光学的に区別しにくい隣接 voxel のみをまとめ、定数 emission
と体積を保存する。wall、inside 境界、partial visibility、急峻な emission を持ち得る
領域は統合しない。mapping、各 group の体積、推定した統合誤差を保存し、結果が何を
表すか追跡可能にする。

最初から octree を導入せず、active voxel compact 化と `res` 自動判定の効果を確認した
後に必要性を判断する。

## 有限サイズ Eye の物理モデル

### Eye 内部位置ごとの局所 etendue

screen 側の etendue はすでに subpixel ごとに `A cos^4(theta) / (4 pi)` を評価し、
projected Eye spot との正確な交差面積を掛けている。一方、source 側の距離と角度補正は
Eye 中心を通る光線で一度だけ評価し、spot 全体で再利用している。

大きい、または細長い Eye については、screen 上の overlap 位置を Eye 内の通過位置へ
逆写像し、位置ごとの距離、角度、etendue を評価する候補がある。wall や aperture による
visibility も Eye 内部で変化する場合は、局所 visibility も同時に扱う必要がある。

実装前に、現在の中心光線近似と局所積分の差を `0.5 x 4 mm` 程度の slit で測り、
必要な分割数と計算量を見積もる。

## Detector と aperture の拡張

### Photodiode array

- active area、pitch、pixel 間の非感光領域を明示できる detector 表現を検討する。
- fill factor が高い配列は連続 screen として扱い、面積積分を利用する。
- active area が spacing より十分小さい疎な配列は、複数の 1x1 screen、または
  multi-detector screen として表す。
- `0.75 x 4 mm`、pitch `0.95 mm`、gap 約 `0.2 mm` のような具体例で検証する。

### Eye 形状

spot の面積積分が直接対応する Eye 形状は circle、ellipse、rectangle である。
任意形状の aperture mesh は遮蔽判定には使えるが、その形状を projected spot の
放射分布として積分する機能とは別である。任意 Eye 形状が必要になった場合は、polygon
clip または数値積分を追加する。

## 意図的に保留している最適化

- `I @ S` の完全融合は保留する。現在の sparse 積は一時的な非ゼロ要素増加を抑えており、
  単純な融合は subvoxel 補間の最大 8 列分を展開して、逆にメモリを増やす可能性がある。
- 全 voxel で一律に高い source res を使う前に、自動判定と適応 res を評価する。
- adaptive voxel merging は、compact mapping だけでは不足すると確認してから着手する。

## 実施順序案

1. cache version と旧 pickle migration
2. `d=75 mm` MST 数値検証と detector res の決定
3. inside 境界仕様の確定
4. main への統合
5. source `res` 自動判定と fully/partial の適応化
6. active voxel compact mapping と `d=10--25 mm` 比較
7. 必要なら adaptive voxel merging
8. 必要性を定量化してから局所 etendue、PD array、任意 Eye 形状へ進む
