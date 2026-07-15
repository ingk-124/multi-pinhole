# Projection matrix roadmap

この文書には、projection matrixの高精度化・高速化について今後実施する項目だけを記録する。
完了した変更の詳細はGit履歴と `docs/ja/world.md` を参照する。

## 現在の前提

- fully visible voxelの外接球からaxis-wise ideal source resを選択できる。
- `adaptive_source_resolution=True`で有限の`res`を指定するとaxis-wise ceiling、`res=None`では
  uncapped idealとなる。
- partial voxelはvisibilityが不連続なのでadaptive化せず、固定`partial_res`を使用する。
- `preflight_projection`でvisibility、res bucket、full sample数、partial sample上限をP構築前に確認できる。
- d=10 wallなしToyでは、uncapped idealに対するcap=5の画像誤差は、滑らかなprofileで約0.2%以下、
  近距離squareで約0.4%、最短距離の単一voxel impulseで約0.9%だった。cap=5は有力候補だが、
  MST実形状で未受け入れでありdefaultにはしない。
- QAによるPSF圧縮とoptical-bin間圧縮はFuture Workへ退避しており、現在の開発対象に含めない。

## 積み残しサマリー

| 優先度 | 項目 | 完了条件 | 依存関係 |
|---|---|---|---|
| P0 | cap=5とpartial resの受け入れ | MST短時間モデルで精度・時間を比較し、production設定を決定 | なし。次の作業対象 |
| P1 | wall visibility / preflight高速化 | d=10でvisibility時間とpeak memoryを削減し、cache再利用を確認 | P0と並行可能 |
| P2 | MST profileとinside境界仕様 | 境界を含むprofileとinverse problem側の未知変数仕様を回帰テストで固定 | P0のreferenceを使用 |
| P3 | Active voxel compact mapping | 列削減、双方向mapping、projection/transpose/pickleを受け入れ | P1、P2の仕様確定後 |
| P4 | Fine-grid最終受け入れ | d=10--25で時間、peak memory、画像誤差、inverse problemを評価 | P0--P3 |
| P5 | Eye内部局所visibility | slit実形状で必要性を定量判定 | 条件付き |

## P0: cap=5とpartial resの受け入れ

### 1. 再現可能なToy benchmark

今回のwallなしd=10検証を `examples/` のスクリプトとして保存し、次を一括出力する。

- uncapped idealとcap=5のpreflight summary、res bucket、source sample数
- P構築時間、peak memory、Pの`nnz`と実byte数
- constant、linear、near square、near Gaussian、nearest impulseの画像誤差
- 総光量差、相対L1/L2、reference peakで規格化した最大pixel誤差

detectorの面積積分とsource quadratureを分離するため、まず`subpixel_resolution=1`を基準とし、
その後`1, 2, 5`でcap=5に関する結論が変わらないことを確認する。

### 2. MST短時間モデル

制限したMST領域またはd=75モデルで、次を同じvisibility cacheから比較する。

1. uncapped ideal reference
2. adaptive cap=5
3. 従来のfixed res=5

profileはinside一定、小半径Gaussian、境界付近まで値を持つprofile、非対称profileを使う。
Toyと同じ画像誤差指標に加えて、projection構築時間、peak memory、camera別画像断面を保存する。
誤差の合格閾値は結果を見て明示的に決め、幾何heuristic自体を画像誤差保証とは扱わない。

### 3. partial res収束

d=10の制限MST preflightでは、`partial_res=5`のsample上限が総sample上限の約44%を占めた。
`partial_res=2, 3, 5`と高精度referenceを比較し、partial境界だけの総光量差とpixel誤差を評価する。
cap=5をfully visible側に導入してもpartial側が律速なら、production設定を別々に決める。

### 4. APIとlegacy cleanup

MST受け入れ後に以下を行う。

- adaptiveをdefaultにするかopt-inのままにするか決定する。
- `res=None`をuncapped idealのまま残すか、推奨ceilingを別引数・presetで表現するか決定する。
- 現在productionから未使用の端点投影方式 `projected_axis_spans`、
  `select_source_resolution`、対応する旧example/testを削除する。
- scalar fixed `res` / `partial_res`を非等方voxelの辺長比でaxis-wise化する必要性を、
  非等方Toyで確認する。adaptive idealのaxis-wise化とは別問題として扱う。

## P1: wall visibilityとpreflightの高速化

制限MST d=10（350×150×70、367.5万voxel）ではpreflightが約448秒かかった。
preflightはpartial subvoxelを評価しておらず、主因はinside頂点とwall meshのvisibility判定である。

1. preflightをvisibility、res選択、集計に分けて時間とpeak memoryを記録する。
2. `stl_utils.check_visible`のcandidate triangle生成、巨大boolean配列、`nonzero`をprofileする。
3. bounded chunk、camera/eye並列、既存cache再利用のうち効果があるものだけを実装する。
4. preflight後の`set_projection_matrix`がvisibilityを再計算しないことを大規模caseで確認する。
5. World保存・読込後にも安全にvisibilityを再利用できる条件を、geometry/cache schemaとともに明記する。

preflight自体を軽い解析とみなさない。wall付きfine gridではvisibility構築も本計算の一部として扱う。

## P2: MST profileとinside境界仕様

source resを上げて改善できるのは、表現済みの三線形補間profile `S f`を積分する誤差だけである。
voxel gridが実際のemissionを表現できない誤差とは分離して評価する。

- inside voxelの補間にoutside側の隣接voxel値が混ざる条件を可視化する。
- outside emissionをゼロ固定するか、projection列から除外するか、inverse problemで拘束するか決める。
- 定数profileの総光量保存、内部一次profile、inside境界を横切るprofileを回帰テストにする。
- aperture/wall境界でpartial res収束が成立することを確認する。

## P3: Active voxel compact mapping

torus外、inside外、または全cameraから常に不可視なvoxelをprojection matrixの列から除外する。

- compact indexと元voxel indexの双方向mappingを保存する。
- emission入力のcompact化、projection、transpose、inverse problem後のgrid復元APIを定義する。
- mapping、visibility、projection列、cacheを含む実byte削減量を測定する。
- constant emission、camera別visibility、pickle roundtripを回帰テストにする。

QA/PSF圧縮を再検討するのは、compact mapping後もPの保存量またはmatvecが律速として残った場合だけとする。

## P4: Fine-grid最終受け入れ

少なくともd=25、12.5、10 mmで以下を比較する。

- preflight visibility時間とres分布
- source sample数、projection構築時間、peak memory、Pの実byte数
- fixed referenceまたは受け入れ済みreferenceに対する画像誤差
- projection/transposeとprofile fittingの実行時間

d=10のfull bounding gridは約1720万voxelになるため、P0--P3を完了してから一度だけ最終benchmarkを行う。

## P5: 条件付き物理拡張

`0.5×4 mm`程度のslitについて、Eye中心visibilityとEye内部位置ごとの局所visibilityの差を実形状で
定量化する。画像差が要求精度に対して無視できない場合だけ、Eye内部visibilityを実装する。

## 次の実装順

1. Toy cap=5 benchmarkを `examples/` に保存し、detector res依存を追加する。
2. 制限MSTでuncapped ideal / cap=5 / fixed 5とpartial resを比較する。
3. production ceiling、partial res、API defaultを決定する。
4. legacy端点投影方式を削除し、adaptive branchをmainへ統合できる状態にする。
5. wall visibilityをprofileし、d=10 preflightを高速化する。
6. inside境界、compact mapping、fine-grid最終benchmarkへ進む。
