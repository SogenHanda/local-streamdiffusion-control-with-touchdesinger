# TouchDesigner monitoring dashboard

`build_monitor_ui.py`は、OSC In DATに届いている`/streamdiffusion/monitor/*`を読み取り、監視GUIを自動生成します。

現在の1列形式に対応しています。

```text
/streamdiffusion/monitor/input_fps 17.93354
/streamdiffusion/monitor/state "実行中"
```

addressとvalueを別カラムへ分割した形式にも対応しています。

## 導入

1. OSC In DATのNetwork Portを`9001`にします。
2. OSC In DATのMaximum Linesは`64`以上にします。すべての監視値を保持するため、`100`を推奨します。
3. 監視GUIを表示するContainerへ`build_monitor_ui.py`をドラッグし、Text DATとして読み込みます。
4. Text DATを右クリックして **Run Script** を実行します。
5. 同じContainer内に`monitor_ui_root`が生成され、Operator Viewerへ自動設定されます。

OSC In DATが同じContainer内またはその子階層にあり、Portが`9001`なら自動検出します。検出できない場合は、スクリプト冒頭へ絶対パスを設定します。

```python
OSC_IN_DAT_PATH = "/project1/control/osc_in_monitor"
```

## OSC In DATの分割設定

スクリーンショットのように全メッセージが`message`列へ入っていれば、そのまま使用できます。`Split Message into Columns`のON/OFFはどちらでも動作します。

値の保持にDAT Executeは使用しません。生成される`_monitor_model`がOSC In DATを新しい行から検索し、各Text COMPへ表示します。

## 再生成

スクリプトを再実行すると`monitor_ui_root`だけを作り直します。OSC In DATや既存のパラメータ・Spout構成には触れません。
