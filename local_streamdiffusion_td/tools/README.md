# TouchDesigner launcher

操作用GUIを自動生成する場合は、[`CONTROL_UI_SETUP.md`](CONTROL_UI_SETUP.md)を参照してください。既存のCustom Parameterへ直接接続するText DATスクリプトは[`build_control_ui.py`](build_control_ui.py)です。

OSC監視GUIを自動生成する場合は、[`MONITOR_UI_SETUP.md`](MONITOR_UI_SETUP.md)を参照してください。OSC In DATの1列`message`形式を直接読み取るスクリプトは[`build_monitor_ui.py`](build_monitor_ui.py)です。

同梱の`local_streamdiffusion_td.toe`には`local_streamdiffusion_launcher.py`がText DATとして埋め込み済みです。ランチャーを更新または別の`.toe`へ導入するときは、このファイルをTouchDesignerへドラッグして読み込み、DAT名を`local_streamdiffusion_launcher`にします。ランチャーは`.toe`と同じ階層の親にある`local_streamdiffusion`を自動検出します。別の場所へ置く場合だけ、環境変数`LOCAL_STREAMDIFFUSION_DIR`で上書きしてください。TensorRTを使わないPCでは`USE_TENSORRT = False`にします。

同梱の`local_streamdiffusion_td.toe`は、OSC address、ランチャーDAT、埋め込みスクリプトを一般名へ移行済みです。

Pythonの監視ウィンドウは既定で表示します。TouchDesignerから送信されたOSC値、推論状態、FPS、GPU情報、ログを確認できます。非表示で運用したい場合だけ、ランチャーの`HIDE_MONITOR = True`または`app.py --hidden`を使用します。

## Runパラメータとの接続

RunがCustom Parameterの場合、Parameter Execute DATの`onValueChange()`へ次を入れます。Parameter Execute DATの監視対象には、Runパラメータを持つCOMPを指定してください。

```python
def onValueChange(par, prev):
    if par.name.lower() == 'run':
        op('local_streamdiffusion_launcher').module.set_run(par.eval())
    return
```

RunがCHOP channelの場合はCHOP Execute DATへ次を入れ、`Value Change`をONにします。

```python
def onValueChange(channel, sampleIndex, val, prev):
    if channel.name.lower() == 'run':
        op('local_streamdiffusion_launcher').module.set_run(val)
    return
```

動作は次の通りです。

- Runが`1`になったとき、Pythonが未起動ならアプリを起動して生成開始
- Pythonが起動済みなら何もせず、二重起動を防止
- Runが`0`になってもPythonと推論を継続

Runが`0`の状態ではアプリを起動せず、実行中にRunが`0`へ戻ってもPythonや推論は停止しません。Runは未起動時に一度だけアプリを起動し、同じPCですでにOSC 13001を使用するアプリがいる場合は二重起動を防止します。

StopはOSC `/streamdiffusion/system/stop`で受け取ります。以前のようにワーカーだけを一時停止せず、Python、モデル、Spout senderを含むアプリ全体を終了します。次のRunではクリーンな単一プロセスとして起動します。

## Stopと終了

TouchDesignerのOSC Outから`/streamdiffusion/system/stop 1`をPulseとして送るか、`op('local_streamdiffusion_launcher').module.stop_generation()`を呼びます。Stop後にRunを押すと新規起動します。`.toe`終了時は`op('local_streamdiffusion_launcher').module.shutdown()`を呼びます。

## .toe起動時の自動起動

Execute DATを1つ作り、`Start`と`Exit`をONにして次を入れます。

```python
def onStart():
    # Python UIを起動し、そのまま生成まで自動開始します。
    run(
        "op('local_streamdiffusion_launcher').module.startup(1)",
        delayFrames=1,
    )
    return


def onExit():
    # .toe終了時に外部Pythonも終了し、次回の二重起動を防ぎます。
    op('local_streamdiffusion_launcher').module.shutdown()
    return
```

起動時のRun値に生成開始を合わせたい場合は、`startup()`へ現在値を渡します。次の`YOUR_CONTROL_COMP`は実際のCOMP名へ置き換えてください。

```python
def onStart():
    run(
        "op('local_streamdiffusion_launcher').module.startup("
        "op('YOUR_CONTROL_COMP').par.Run.eval())",
        delayFrames=1,
    )
    return
```

`HIDE_MONITOR = False`でPython画面を表示した場合、ログへ次が表示されれば接続準備完了です。

```text
OSC受信を開始しました: udp://127.0.0.1:13001
```
