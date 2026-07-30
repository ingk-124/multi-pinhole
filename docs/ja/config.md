# World config schema

> **Level 3 — reference。** これはkey・型・errorを網羅する仕様書です。最初から
> 通読せず、まず[World JSON設定ガイド](world-config-guide.md)でfileを作り、
> 正確な契約を確認する時だけ参照してください。

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

`voxel.type`で排他的なgrid表現を1つ選びます。

- `"uniform"`は3組の`[minimum, maximum]`を持つ`ranges`と
  `shape = [N_x, N_y, N_z]`が必須です。各axisに`N_axis + 1`個の等間隔な
  boundaryを生成します。
- `"axes"`は狭義単調増加なboundary array `x`, `y`, `z`を持つ`axes`が必須です。
  shapeとrangeはaxisから導出します。

2形式は混在できません。uniformに`axes`、axes形式に`ranges`/`shape`を指定すると
errorです。`to_config`は3軸がすべて等間隔ならcompactなuniform形式、それ以外は
explicit axes形式を書きます。`sub_voxel_resolution`は3つの正整数です。

`coordinate`は`type`と数値`parameters`を持ちます。座標名とparameterは`Voxel`と
同じものを使えます。Voxelは回転済み座標frameを永続化しません。単発の変換には
座標変換helperの`rotation`引数を使い、物理sceneはworld座標上で整列させます。

## Cameraと光学系

各cameraは次を持ちます。

- stringの`key`と`name`
- mm単位のWorld `position`と、次のいずれか1つのorientation表現
  - 3×3のworld-to-camera `rotation_matrix`
  - world座標の`look_point`と、world座標の`right_point`または
    `down_point`のいずれか1つを持つ`orientation`
- 1つ以上のEye。`type`、2次元`position`、`focal_length`、解析的`shape`、
  2次元`size`、`wavelength_range`
- Screen。解析的`shape`、2次元`size`、正整数2次元`pixel_shape`、
  正の`subpixel_resolution`
- 解析的Apertureのarray。version 1は`circle`、`ellipse`、`rectangle`と
  `position`、`direction`を扱います。任意の正整数`resolution`は境界分割数
  （既定値`20`）、`max_size`は`null`またはsupport meshの正の2方向半幅です。
- path付きSTL Aperture。`type`、`path`、camera座標の`position`を持ちます。
  STL頂点はcamera座標内であらかじめ所望の向きになっている必要があります。

pointによるorientationは`Camera.set_orientation_from_points()`でmatrixへ変換します。
pointは方向vectorではなく絶対world座標です。orientation表現を両方またはどちらも
指定すること、`right_point`と`down_point`を両方指定することはerrorです。

source pathなしのin-memory STL objectから作ったApertureをJSONへ書くことはできず、
`to_config`は`WorldConfigError`を送出します。

## Pathとinside mask

wallとSTL Apertureのpathはabsoluteでも構いません。relative pathはJSON fileの
directory基準で、mapping入力ではcurrent directory基準です。読み込んだconfigを
別の場所へ書くと、`to_config`は新config directoryからのrelative pathを書きます。
mesh objectから直接組み立てたWorldには信頼できるsource pathがないため、
roundtrip可能と偽らず失敗します。

version 1のinside typeは次の4つです。

- `{"type": "all", "parameters": {}}`
- `{"type": "box", "parameters": {"ranges": [[xmin, xmax], ...]}}`
- `{"type": "sphere", "parameters": {"center": [x, y, z], "radius": r}}`
- `{"type": "torus", "parameters": {"major_radius": R0, "minor_radius": a}}`
  （`(sqrt(x^2 + y^2) - R0)^2 + z^2 <= a^2`）

任意Python callableと単独の`inside_vertices` arrayは非対応です。application codeで
構築するか、Python stateの正確なcheckpointが必要ならWorld archiveを使います。

## Errorとversion

不正入力は`ValueError`のsubclassである
`multi_pinhole.config.WorldConfigError`を送出します。非対応schema versionは明示的に
失敗します。将来のconfig schema更新はpackage version、World archive schemaの
どちらからも独立しています。
