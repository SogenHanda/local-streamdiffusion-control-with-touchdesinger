# TouchDesigner launcher

`local_inference_launcher.py`をTouchDesignerへドラッグしてText DATとして読み込み、DAT名を`local_inference_launcher`にします。別PCではText DAT上部の`APP_DIR`を、そのPCの`local_inference`フォルダへ変更してください。TensorRTを使わないPCでは`USE_TENSORRT = False`にします。

## Runパラメータとの接続

RunがCustom Parameterの場合、Parameter Execute DATの`onValueChange()`へ次を入れます。Parameter Execute DATの監視対象には、Runパラメータを持つCOMPを指定してください。

```python
def onValueChange(par, prev):
    if par.name.lower() == 'run':
        op('local_inference_launcher').module.set_run(par.eval())
    return
```

RunがCHOP channelの場合はCHOP Execute DATへ次を入れ、`Value Change`をONにします。

```python
def onValueChange(channel, sampleIndex, val, prev):
    if channel.name.lower() == 'run':
        op('local_inference_launcher').module.set_run(val)
    return
```

動作は次の通りです。

- Runが`1`になったとき、Pythonが未起動ならアプリを起動して生成開始
- Pythonが起動済みならOSC `/ergonomics/system/start`を送信
- Runが`0`になったとき、Python UIは残してOSC `/ergonomics/system/stop`を送信

Runが`0`でもPython UI自体は残ります。ただし現在のStop処理はGPU上のモデルを解放するため、Runを再び`1`にするとモデル読込が発生します。

## .toe起動時の自動起動

Execute DATを1つ作り、`Start`と`Exit`をONにして次を入れます。

```python
def onStart():
    # Python UIを起動し、そのまま生成まで自動開始します。
    run(
        "op('local_inference_launcher').module.startup(1)",
        delayFrames=1,
    )
    return


def onExit():
    # .toe終了時に外部Pythonも終了し、次回の二重起動を防ぎます。
    op('local_inference_launcher').module.shutdown()
    return
```

起動時のRun値に生成開始を合わせたい場合は、`startup()`へ現在値を渡します。次の`YOUR_CONTROL_COMP`は実際のCOMP名へ置き換えてください。

```python
def onStart():
    run(
        "op('local_inference_launcher').module.startup("
        "op('YOUR_CONTROL_COMP').par.Run.eval())",
        delayFrames=1,
    )
    return
```

Python起動後、Python UIのログへ次が表示されれば接続準備完了です。

```text
OSC受信を開始しました: udp://127.0.0.1:13001
```
