# 0.8/0.9 Worldから1.0への移行

## 公開import

次を使います。

```python
from multi_pinhole import (
    Aperture,
    Camera,
    Eye,
    Rays,
    Screen,
    Voxel,
    World,
)
```

`multi_pinhole.__version__`は正式な公開値です。`multi_pinhole.core`は旧importと
pickle/dillのclass global解決専用に残します。private rasterizer helper、
`stl_utils`、旧type aliasはexportしません。内部開発で本当に必要なhelperは、
その定義moduleからimportしてください。

## 信頼できる保存済みWorldの移行

```python
from multi_pinhole import World

world = World.load("old-world.pkl")
world.save("world-v1.mpw")
metadata = World.inspect_archive("world-v1.mpw")
assert metadata["world_schema_version"] == 1
assert metadata["projection_cache_schema_version"] == 3
```

loaderは旧`multi_pinhole.core.*` class globalを解決します。互換なschema 3の
visibilityとprojection cacheを維持します。projection cache schemaがない、または
非互換な場合は可能な限りvisibilityを残し、projectionだけを消去します。代表的な
`world.project(emission, key)`結果を移行前後で比較してください。

direct dillとarchive payloadは任意codeを実行できるため、信頼できる旧fileだけを
読み込んでください。

## Configとarchiveの選択

review可能なscene構築にはJSON config、完全なPython stateとcacheには`.mpw`を
使います。configは解析的Aperture、source pathのあるSTL wall、built-inの`all`、
`box`、`sphere` inside maskを扱います。任意callable、単独inside array、
STL Aperture、source provenanceのないwall meshは、黙って欠落させず拒否します。

1.0ではoptics moduleを移動していません。projection数式、処理順、dtype、許容誤差、
projection cache schema 3は変更していません。

