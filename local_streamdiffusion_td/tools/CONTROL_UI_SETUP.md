# TouchDesigner control dashboard

`build_control_ui.py`は、既存のCustom Parameterをそのまま操作するダッシュボードを自動生成します。値を複製せず、TouchDesigner標準のParameter COMPを使うため、既存のOSC送信・Parameter Execute DAT・Pulse処理を変更する必要はありません。

## 導入

1. スクリーンショットのCustom Parameterを持つContainer COMPへ、`build_control_ui.py`をドラッグしてText DATとして読み込みます。
2. Text DATを右クリックして **Run Script** を実行します。
3. 同じContainer COMP内に`ui_root`が生成され、そのContainerのOperator Viewerに自動設定されます。
4. Container ViewerをActiveにすると、そのまま操作できます。

スクリプトを再実行した場合は`ui_root`だけを作り直します。既存のCustom Parameter、OSC DAT、Execute DAT、Spout構成には触れません。

## 前提となるCustom Page名

スクリプトは次の3ページを表示します。

- `AI setting parameter`
- `realtime parameter`
- `osc setting`

実際のページ名が違う場合は、スクリプト冒頭の`PAGE_AI`、`PAGE_LIVE`、`PAGE_OSC`を変更してから再実行してください。

Text DATを別の場所へ置く場合は、`SOURCE_OP_PATH`へCustom Parameterを持つCOMPの絶対パスを設定します。

```python
SOURCE_OP_PATH = "/project1/streamdiffusion_control"
```

## レイアウト意図

- 左: 本番中に触るRealtime Parameter
- 中央: モデル、TensorRT、Performance、Resolution、Spout、Apply/Run
- 右上: OSC接続
- 右下: 本番時の操作順

モデル再読込を伴う設定とリアルタイム値を視覚的に分離し、誤操作を減らす構成です。色はRealtimeを緑、再初期化を伴うAI設定を黄で区別しています。

## サイズ変更

標準サイズは`1366 x 768`です。変更する場合は、スクリプト冒頭の`UI_WIDTH`と`UI_HEIGHT`だけでなく、各カードの座標も合わせて変更してください。現状は本番用の16:9画面に合わせた固定レイアウトです。
