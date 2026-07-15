# QA / PSF圧縮 Future Work

この文書は、projection matrixのPSF類似性を利用した圧縮案を、再検討できる形で保存する。
現行productionはnativeな疎行列 `P` を構築し、この圧縮は使用しない。

## 記号と狙い

1台のcamera・1つのEye・1つの処理scopeについて、

- `N_p`: detector pixel数
- `N_s`: 可視subvoxel sample数
- `N_v`: voxel数
- `I` (`N_p × N_s`): 各subvoxel sampleのpixel上のインパルス応答（PSF）
- `S` (`N_s × N_v`): voxel中心値からsample値への三線形補間とsample体積重み
- `P = I S` (`N_p × N_v`): 保存するprojection matrix

とする。似たPSF列をまとめ、`I ≈ Q R`、したがって

```text
P = I S ≈ Q R S = Q A
A = R S
```

と因子化する。`Q`は代表PSFまたは基底、`R`はsampleを代表PSFへ写す係数、`A`はvoxel値を
代表PSFの振幅へ写す行列である。grouping方式の`R`は各列がone-hotに近い短い係数になる。
これは列クラスタリングまたは低ランク近似であり、`Q`の直交性を要求するQR分解ではない。
無圧縮は概念的に `Q=I, R=identity, A=S` である。

## scopeとchunk

1. world座標の可視sampleをEye座標へ変換する。
2. `xi=X/Z`, `eta=Y/Z`（実装では `Eye.calc_rays` と一致する規約）をdetector pixel pitchでbinningする。
3. 同一optical bin内を近いPSFの候補scopeとする。必要ならzoom/etendue差を抑えるためZ方向も分割する。
4. optical scopeは圧縮の境界、work chunkはmemory上の処理単位とする。複数scopeを1 chunkへ詰めても、
   初期実装ではscopeをまたいで圧縮しない。
5. `P_allcam`を一括因子化せず、camera/Eye単位で処理する。

detector pixel行もscopeの投影support付近だけ抜き出せば、局所的なdense `I_scope`を小さく保てる。
`S`全体は保存せず、chunk内で `A_c=R_c S_c` を作る。

## 候補アルゴリズム

- **farthest / recursive**: 加重平均PSFを代表とし、最大距離の列を使って再帰的に分割する。
- **leader / one-pass**: 列を走査し、既存代表との距離がtolerance以下なら所属、なければ新groupを作る。
  高速だが順序依存と代表PSFのdriftがあるため、固定代表、medoid、または誤差上限の再検査が必要。
- 距離はkeywordで `relative_l2` と `l1` を選択可能にする。L2は計算しやすく、L1は非負画像の
  光量誤差と結び付けやすい。
- `tolerance == 0` は明示的に圧縮処理を迂回し、native `P` を構築する。

近距離だけを特別扱いせず、必要なgroup数がtoleranceによって自然に増える設計を先に試す。
圧縮効果はsource resと `Z/f` がともに大きい場合に期待される。adaptive resが通常1--5で済むなら、
因子管理の複雑さに見合わない可能性が高い。

## 保存形式と受け入れ判定

- membershipはboolean行列ではなく整数 `group_index` と係数で保持する。
- 最終判断は `bytes(Q)+bytes(A)` とnative sparse `bytes(P)` の比較で行う。`nnz`は疎性の診断には
  使えるが、index配列とdtypeを含む実memoryの代用にはしない。
- factor bytesがnative `P`の指定比率（検討時の案はdefault 0.8）を下回らなければnative `P`を採用する。
- factorized operatorを公開する場合は `.project()` / `.dot()` / `.transpose()` を定義し、単なる
  scipy sparse matrixと誤認させない。

## 誤差検証

最初に同一voxel内の `res^3` PSFを圧縮し、その後にvoxel間の共有を検討する。段階ごとのtoleranceだけでなく、
最終的な `Q A` とnative `P`を直接比較して誤差の蓄積を確認する。

- PSFのshapeを比較する前に、振幅（局所etendue/総感度）と正規化shapeを分離する。
- constant、linear、square、Gaussian、single impulse、および符号付きprofileで画像L1/L2・最大誤差・総光量を比較する。
- 定数profileの総光量保存と、三線形補間された内部一次profileを回帰条件に含める。
- projectionとtransposeの両方を検証する。

## 再開条件

wall visibility高速化とactive-voxel compact mappingの後でも、`P`の保存量またはmatvecが実際の律速として残り、
toy benchmarkでfactor bytesがnative `P`を十分下回る場合だけproduction化を再検討する。
現時点の実験コードは `benchmarks/evaluate_projection_column_compression.py` と
`benchmarks/evaluate_subvoxel_psf_compression.py` に保存する。
