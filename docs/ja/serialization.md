# World archiveとserialization契約

完全なcheckpointには`world.save("scene.mpw")`、復元には
`World.load("scene.mpw")`を使います。`World.inspect_archive(path)`はWorldを
unpickleせずmetadataを読みます。

## Containerとmanifest

`.mpw`は次のZIP archiveです。

```text
scene.mpw
├── manifest.json
└── world.pkl
```

`manifest.json`は次のUTF-8 JSONです。

```json
{
  "format": "multi-pinhole-world",
  "world_schema_version": 1,
  "library_version": "1.0.0",
  "projection_cache_schema_version": 3,
  "python_version": "...",
  "numpy_version": "...",
  "scipy_version": "..."
}
```

3つのversion軸は独立です。

- `library_version`はarchiveを書いたpackageを示します。
- `world_schema_version`はWorld payload/container契約を管理します。
- `projection_cache_schema_version`は行列の意味、shape、ordering、保存形式を
  管理します。

module移動だけではprojection cache schemaを変更しません。1.0では数値表現が
変わらないためschema 3を維持します。

## 読み込みとcache migration

World schema 1は通常どおり読み込みます。非対応World schemaはpayloadをunpickleする
前に失敗します。projection cache schemaが3なら、visibility、eye別projection、
projection settings、集約済み`P_matrix`を維持します。非互換なら可能な限り
visibilityを残し、projection dataだけを安全な再計算のために消去します。

`World.load`は旧direct-dill fileも検出します。manifestがないため、
`inspect_archive`は`format="legacy-direct-dill"`とversion不明を返します。
信頼できる旧fileを読み込んだ後、`world.save("migrated.mpw")`で移行します。
互換aliasの`save_world`と`load_world`は新writerと自動判別loaderを使います。

保存はdestination directory内の一時fileへ書いた後、atomic replaceします。
live Worldのcacheをnormalizeまたは無効化しません。

## Security境界

JSON manifestはdataとして安全に確認できます。`World.load`は`world.pkl`または
legacy fileからpickle/dill復元を実行します。pickle/dillは任意codeを実行できます。
信頼できない、または真正性を確認できないarchive／legacy Worldを決して
読み込まないでください。

cacheや任意Python stateが不要な交換用途には、callableを実行しないJSON
[World config schema](config.md)を使ってください。
