# multi-pinhole ドキュメント

必要な理解の深さに応じて3層に分けています。ファイル名順に読む必要はありません。

## Level 1 — ライブラリを使う

内部実装を理解せず、cameraを定義して投影し、結果を確認したい場合はここだけで
十分です。

1. [概要と最初の投影](overview.md)
2. [World JSONの組み立て方](world-config-guide.md)
3. [座標・profile・補間](coordinates-profiles.md)
4. [voxel分布と検出器画像の可視化](visualization.md)

実行可能なcodeは[`examples/`](../../examples)にあります。JSONからsceneを作る
一連の例は[`examples/relax`](../../examples/relax/README.md)が最も完全です。

> **通常利用ならLevel 1までで十分です。**
> `World`, `Voxel`, `Camera`, `World.from_config`,
> `World.set_projection_matrix`, `World.project`が基本workflowを担います。

## Level 2 — 科学モデルを理解する

数値resolutionを決める、単位や座標規約を確認する、論文のmethodを書く、結果の
妥当性を調べる場合に読みます。

- [pinhole opticsと検出器モデル](core.md) — 座標系、pinhole投影、
  aperture/eye、有限eyeとdetector積分。
- [Worldの投影モデル](world.md) — visibility、source積分、部分可視voxel、
  adaptive resolution、projection/backprojection。
- [座標・profile・補間](coordinates-profiles.md)の後半 — angle規約、
  normalization scale、補間の限界。

各ページは利用者に見える契約から始まります。**内部実装**と明記した節は、
algorithmの検証・改修をしない限り読み飛ばせます。

## Level 3 — Reference・保存・開発

- [JSON schema reference](config.md) — key、型、errorの網羅的仕様。
- [World archiveとserialization](serialization.md) — cache保存、互換性、
  pickleのsecurity境界。
- [utilityとgeometry内部](utilities.md) — STL交差判定と低水準helper。
- [1.0へのmigration](migration-v1.md) — 古いWorldとimport path。
- [開発・release方針](development-roadmap.md)。

roadmap文書は将来計画であり、現在の利用方法ではありません。

## 新しい説明をどこへ置くか

情報を分散させないため、次の原則を使います。

- 手順は`overview`、`world-config-guide`、`visualization`
- 科学的仮定・数式・近似は`core`または`world`
- keyやsignatureの網羅的仕様はreference
- private algorithmと性能上の工夫は対応する科学ページの後半

[English documentation](../README.md)
