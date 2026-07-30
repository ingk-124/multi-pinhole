# World config schema

`World.from_config(path_or_mapping)`は
`multi-pinhole/world-config` schema version 1を読み込みます。
`world.to_config(path)`は同じcanonical JSONを書き、mappingも返します。
configは宣言的なscene記述であり、visibility、eye別projection、`P_matrix`、
cache schemaを保存しません。

## Root契約

root fieldはすべて必須です。

| Field | 契約 |
| --- | --- |
| `schema` | 正確に`multi-pinhole/world-config` |
| `schema_version` | 整数`1` |
| `units` | 正確に`{"length": "mm", "angle": "rad"}` |
| `voxel` | 下記のVoxel object |
| `cameras` | 順序付きarray。`key`は一意なstring |
| `walls` | `{"type": "stl", "path": "..."}`のarray |
| `inside` | `null`または安全なbuilt-in inside mask |
| `verbose` | 整数 |

schemaの全階層で未知fieldと必須field不足をerrorにします。数値は有限でなければ
なりません。JSONにPython import、式、callableを記述できず、loaderは`eval`を
使いません。

## Voxel

`voxel.axes`は2要素以上の狭義単調増加`x`、`y`、`z` arrayです。`shape`は各axisの
長さから1を引いた値、`ranges`はaxis端点と一致しなければなりません。
`sub_voxel_resolution`は3つの正整数です。

`coordinate`は`type`、数値`parameters`、3×3のWorld回転行列を持ちます。
行列は直交し、行列式が+1でなければなりません。座標名とparameterは`Voxel`と
同じものを使えます。axes、ranges、shapeの冗長性は意図的です。不一致時に一方を
黙って採用せず、壊れたconfigとして扱います。

## Cameraと光学系

各cameraは次を持ちます。

- stringの`key`と`name`
- mm単位のWorld `position`と3×3のworld-to-camera `rotation_matrix`
- 1つ以上のEye。`type`、2次元`position`、`focal_length`、解析的`shape`、
  2次元`size`、`wavelength_range`
- Screen。解析的`shape`、2次元`size`、正整数2次元`pixel_shape`、
  正の`subpixel_resolution`
- 解析的Apertureのarray。version 1は`circle`、`ellipse`、`rectangle`と
  `position`、`direction`を扱います。

任意STL objectから作ったApertureは解析的Apertureとして表現できないため、
`to_config`は`WorldConfigError`を送出します。

## Pathとinside mask

wall pathはabsoluteでも構いません。relative pathはJSON fileのdirectory基準で、
mapping入力ではcurrent directory基準です。読み込んだconfigを別の場所へ書くと、
`to_config`は新config directoryからのrelative pathを書きます。mesh objectから
直接組み立てたWorldには信頼できるsource pathがないため、roundtrip可能と偽らず
失敗します。

version 1のinside typeは次の3つです。

- `{"type": "all", "parameters": {}}`
- `{"type": "box", "parameters": {"ranges": [[xmin, xmax], ...]}}`
- `{"type": "sphere", "parameters": {"center": [x, y, z], "radius": r}}`

任意Python callableと単独の`inside_vertices` arrayは非対応です。application codeで
構築するか、Python stateの正確なcheckpointが必要ならWorld archiveを使います。

## Errorとversion

不正入力は`ValueError`のsubclassである
`multi_pinhole.config.WorldConfigError`を送出します。非対応schema versionは明示的に
失敗します。将来のconfig schema更新はpackage version、World archive schemaの
どちらからも独立しています。
