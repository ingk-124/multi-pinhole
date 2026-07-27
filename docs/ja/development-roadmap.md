# 開発・リリース方針

この文書は、`multi-pinhole` の公開API、パッケージ構造、互換性、
品質管理に関する中期的な方針をまとめる。投影行列の数値検証や性能改善は
[`projection-roadmap.md`](projection-roadmap.md)、将来のQA／PSF圧縮は
[`projection-compression-future.md`](projection-compression-future.md)を
参照する。

現在のパッケージバージョンは0.8系である。以下のバージョン番号は作業範囲を
整理するための目安であり、日程やリリースを保証するものではない。

## 基本方針

- Python 3.10以上をサポートする。
- 利用者向けの標準入口は `from multi_pinhole import ...` とする。
- 現在の利用者は開発者本人のみである。1.0以前は、内部モジュールのimport経路を
  外部利用者向けに長期間維持することより、保存済みWorldと計算済みcacheを失わない
  ことを優先する。
- 保存済みWorld、標準`pickle`／`dill`、visibility cache、projection matrixを
  読み込み、可能な範囲で再利用できる状態を維持する。
- 科学計算で定着した `N_pixel` や `P_matrix` などの名前は、snake_caseだけを
  理由に変更しない。
- `World`はシーン状態とキャッシュを所有し、数値計算用のprivateモジュールは
  Worldの状態を所有しない。
- `pyproject.toml`を実行時依存関係の正本とし、`requirements.txt`には同じ
  直接依存関係を記載する。

## 0.8系: 1.0に向けた安定化

0.8系では、既存のパッケージ構造と公開APIを維持したまま品質基盤を整える。

- PEP 8、PEP 257、PEP 484に関する明確な問題を修正する。
- 公開APIのdocstringに用途、引数、戻り値、利用者が対処できる例外、配列shape、
  物理単位を記載する。
- 内部コメントは、アルゴリズムの理由、不変条件、数値的仮定、メモリ制御など、
  コードだけでは意図を判断しにくい箇所へ限定する。
- Ruffは最初に `E4`、`E7`、`E9`、`F` のみを必須とし、規則を段階的に追加する。
- formatterの行長は88文字を原則とし、数式、URL、テストデータ、診断メッセージ
  などは可読性を優先して120文字まで許容する。
- 公開API、座標変換、投影と逆投影、可視性、シリアライズの回帰テストを維持する。

この段階では、ファイル移動や公開クラスの正規モジュールパス変更を品質修正と
同じ差分に混ぜない。

## 0.9.0: `multi_pinhole.optics` 候補

光学系モジュールがさらに増える場合、1.0候補として次のサブパッケージを導入する。

```text
multi_pinhole/
├── optics/
│   ├── __init__.py
│   ├── aperture.py
│   ├── camera.py
│   ├── eye.py
│   ├── rays.py
│   └── screen.py
├── projection.py
├── voxel.py
└── world.py
```

この変更は必須ではない。現在の責務分割で十分に保守できる限り、見た目の整理だけを
目的とした移動は行わない。導入する場合は、次を満たす独立した変更として扱う。

- `from multi_pinhole import Camera, Eye, Screen, Aperture, Rays`を正式入口として維持する。
- `multi_pinhole.optics`からも同じクラスオブジェクトを公開する。
- `multi_pinhole.camera`、`eye`、`screen`、`aperture`、`rays`の旧経路を
  過去のpickleを読み込むための軽量なshimとして残す。一般利用者向けの長期的な
  非推奨期間を設けること自体は目的としない。
- `multi_pinhole.core`も、過去のpickleまたは保存済みWorldが参照する間は
  最小限の読み込み互換facadeとして残す。
- 新旧importから得たクラスが同一オブジェクトであることをテストする。
- 0.8系でvisibilityとprojection matrixを計算済みの代表的なWorldを保存し、
  標準`pickle`／`dill`の読み込み互換fixtureとして管理する。
- fixtureを読み込んだ直後にvisibilityとprojection cacheが不必要に無効化されず、
  投影結果を再現できることをテストする。
- 投影キャッシュのschema互換性はPythonクラスのpickle互換性と分けて検証する。

クラスの `__module__` が変わると、新しく作成したpickleを旧バージョンで読めなくなる。
そのため、実装を移動する前に正規モジュールパスとシリアライズ方針を決定する。
旧形式を恒久的に維持する必要がない場合でも、既存データを一度読み込み、新形式で
保存し直すまでshimを削除しない。必要に応じて、この読み込みと再保存だけを行う
小さな移行スクリプトを用意する。

保存済みWorldの移行では、次を別々に確認する。

1. pickle/dillが旧クラスパスを解決してWorldを復元できること。
2. projection cache schemaが互換であり、visibility、eyeごとのprojection、
   集約済み`P_matrix`を再利用できること。
3. 同じemissionに対する投影結果が移行前後で一致すること。

モジュール移動だけでprojection cacheの表現が変わらない場合、cache schemaを
機械的に更新しない。行列の意味、shape、ordering、保存形式が変わる場合にのみ
schemaを更新し、安全に再計算へフォールバックさせる。

## 1.0.0: 公開契約の確定

1.0.0は、単にバージョン番号を上げるのではなく、以下を正式な保守対象として
確定できた段階でリリースする。

- トップレベル公開APIと `__all__`
- サポートするPython、NumPy、SciPyの範囲
- 配列shape、座標系、物理単位に関する公開契約
- import経路とシリアライズ互換性
- 既存の保存済みWorldと計算済みcacheを新しい正規形式へ移行できること
- Worldおよび投影キャッシュのversion/schema方針
- Ruff、formatter、テスト、パッケージbuildの品質ゲート
- 主要examplesが正式な公開APIだけで実行できること
- 0.8／0.9系からの移行方法

`optics`を導入する場合は0.9系で十分に検証してから1.0.0で正式化する。
検証が不十分な場合は、構造変更を1.1.0以降へ延期し、既存構造のまま1.0.0を
リリースしてよい。

## 1.0以降のversioning

Semantic Versioningを次のように適用する。

- **Patch（1.0.1）**: 後方互換なバグ修正、docstring、型注釈、内部最適化。
- **Minor（1.1.0）**: 後方互換な機能追加、新しい公開APIやサブパッケージ。
- **Major（2.0.0）**: 公開API、import経路、保存形式の意図的な非互換変更。

旧import shimは、既存の保存済みWorldを新形式へ移行し終えるまでは削除しない。
移行後は、保守上不要であれば次の明示的なversion更新で削除してよい。多数の
外部利用者を前提とした長い非推奨期間は必須としないが、削除時にはpickleとcacheへの
影響をrelease noteへ記載する。

## 変更時の確認事項

変更ごとに次を確認する。

1. 数値結果、配列shape、座標系、単位が変わるか。
2. 保存済みWorld、pickle／dill、visibility／projection cacheへ影響するか。
3. Worldまたは投影キャッシュのschema更新が必要か。
4. 型注釈、docstring、examplesも同時に更新したか。
5. 不具合または新しい契約を固定する回帰テストを追加したか。
6. benchmarksの測定条件と結果解釈を維持できているか。

最低限、次の検証を実行する。

```bash
python -m compileall -q multi_pinhole tests examples benchmarks
pytest -q
python -m ruff check .
python -m ruff format --check .
git diff --check
git status --short
```

自動修正やformatterを実行した場合は、動作変更が混入していないか差分を確認する。
