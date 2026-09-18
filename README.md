# Local StreamDiffusion + TouchDesigner

WindowsとNVIDIA RTX GPU上で、TouchDesignerのカメラ映像をSpout経由で
ローカルのStreamDiffusionへ送り、生成映像を再びTouchDesignerへ戻す
リアルタイム画像生成システムです。

クラウドAPIは使用しません。初回セットアップとモデル取得後は、インターネットに
接続していない環境でも実行できます。

別PCのCodexへ開発を引き継ぐ場合は、先に
[CODEX_HANDOFF.md](CODEX_HANDOFF.md)を読ませてください。Codex向けの作業ルールは
[AGENTS.md](AGENTS.md)にあります。

## リポジトリ構成

```text
local_streamdiffusion/       Python / CUDA / StreamDiffusion推論
local_streamdiffusion_td/    TouchDesignerプロジェクトと補助スクリプト
```

主な通信設定は次のとおりです。

| 用途 | 設定 |
|---|---|
| 映像入力 | Spout `TD_Camera` |
| 映像出力 | Spout `AI_Output` |
| TD → Python制御 | OSC UDP `127.0.0.1:13001` |
| Python → TD監視 | OSC UDP `127.0.0.1:9001` |
| OSC名前空間 | `/streamdiffusion` |

## 動作環境

- Windows 10 / 11 64-bit
- NVIDIA RTX GPUと最新のNVIDIAドライバー
- Python 3.10 64-bit
- Git for Windows
- TouchDesigner 2023または2025
- 初回セットアップ時のみインターネット接続

RTX 4070ti以上のGPUを搭載しているPCを想定しています。Python 3.11などが
すでに入っていても削除する必要はありません。Python 3.10を追加でインストールし、
このプロジェクト専用の仮想環境として共存させます。

インストール前に確認します。

```powershell
git --version
py -3.10 --version
nvidia-smi
```

`py -3.10`が見つからない場合は、Python 3.10.11 64-bitを追加インストールして
ください。インストーラーでは`Add python.exe to PATH`と`Python Launcher`を
有効にします。

## 新しいPCへの初回セットアップ

### 1. リポジトリをcloneする

初回clone時はGit Credential Managerのブラウザ認証が表示される場合があります。

```powershell
cd "C:\任意の保存先"
git clone https://github.com/SogenHanda/local-streamdiffusion-control-with-touchdesinger.git
cd .\local-streamdiffusion-control-with-touchdesinger
```

フォルダ構成は変更せず、`local_streamdiffusion`と
`local_streamdiffusion_td`を同じ親フォルダに置いてください。TouchDesignerの
`tools/local_streamdiffusion_launcher.py`は、この兄弟関係からPython側の場所を
自動検出します。`.toe`内の埋め込みランチャーに別PCの絶対パスが残っている場合は、
初回だけ`tools`版の内容へ差し替えるか、そのPCの`APP_DIR`へ変更してください。

### 2. 標準環境と基本モデルを導入する

PowerShellで次を実行します。

```powershell
cd .\local_streamdiffusion
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

`-Scope Process`なので変更は現在のPowerShellだけに適用され、Windows全体の
実行ポリシーは変更しません。ポリシーエラーが続く場合は次でも実行できます。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

この処理で以下が導入されます。

- `.venv` Python 3.10仮想環境
- CUDA 12.1版PyTorch 2.1.0
- xFormers / StreamDiffusion / Diffusers / SpoutGL
- DreamShaper 8
- SD 1.5用LCM-LoRA
- TinyVAE（TAESD）

モデルを後で取得したい場合は、ライブラリだけを導入できます。

```powershell
.\setup.ps1 -SkipModels
.\download_models.ps1
```

### 3. 追加モデルを取得する

基本モデル以外を試す場合だけ実行します。

```powershell
.\download_models.ps1 --preset absolute-reality-1.81
.\download_models.ps1 --preset realistic-vision-v5.1
.\download_models.ps1 --preset epicrealism
.\download_models.ps1 --preset lcm-dreamshaper-v7
.\download_models.ps1 --preset sd-turbo
```

利用可能なpreset一覧:

```powershell
.\.venv\Scripts\python.exe .\download_models.py --list-presets
```

すべてのモデルをまとめて取得する場合:

```powershell
$modelPresets = @(
    "dreamshaper-8",
    "absolute-reality-1.81",
    "realistic-vision-v5.1",
    "epicrealism",
    "lcm-dreamshaper-v7",
    "sd-turbo"
)

foreach ($modelPreset in $modelPresets) {
    .\download_models.ps1 --preset $modelPreset
}
```

モデルごとのライセンスと利用条件は、使用前に各Hugging Face model cardで
確認してください。

### 4. 標準版で動作確認する

TensorRTを使う前に標準環境を確認できます。

```powershell
.\run.cmd
```

Pythonの監視画面が開き、OSC受信開始ログが表示されれば起動成功です。生成自体は
TouchDesignerからRunを送るか、次のように自動開始します。

```powershell
.\run.cmd --autostart
```

TouchDesigner内のランチャーは初期状態でTensorRT環境を使用します。標準環境だけで
運用する場合は、Text DAT `local_streamdiffusion_launcher`の
`USE_TENSORRT = False`へ変更してください。

## TensorRT高速化セットアップ（推奨）

TensorRTは標準`.venv`を変更せず、専用の`.venv-trt`へ導入します。標準環境と
基本モデルのセットアップが完了してから実行してください。

```powershell
cd "C:\任意の保存先\local-streamdiffusion-control-with-touchdesinger\local_streamdiffusion"
Set-ExecutionPolicy -Scope Process Bypass
.\setup_tensorrt.ps1
```

TensorRT環境、依存ライブラリ、1モデル分のエンジンを含めて15GB以上の空き容量を
確保してください。

TensorRTエンジンは、次の組み合わせごとに作成が必要です。

- 使用するGPU
- モデル
- 解像度
- 1-step / 2-step
- LCM-LoRAの有無
- TensorRT / CUDAランタイム

別GPUで作ったエンジンはコピーせず、使用するPC上で再ビルドしてください。

### 初回エンジン作成

1. TouchDesigner側のBackendを`Auto`または`xFormers`にします。
2. 使用するモデル、解像度、PerformanceをTouchDesignerから送信します。
3. Python側へ設定が反映・保存されたことを確認してStopします。
4. PowerShellで次を実行します。

```powershell
.\build_tensorrt_engine.cmd
```

ビルドには10〜30分程度かかる場合があります。完了後、TouchDesigner側のBackendを
`Auto`または`TensorRT FP16`へ変更してRunします。

設定済みの`config.json`がない場合は、DreamShaper 8 / 512×512 / 1-stepの
既定値でビルドされます。設定ファイルを手動で準備する場合は次のようにします。

```powershell
Copy-Item .\config.example.json .\config.json
```

`config.json`を編集してから`build_tensorrt_engine.cmd`を実行してください。

TensorRT版をPython側から直接起動する場合:

```powershell
.\run_tensorrt.cmd
```

## TouchDesignerのセットアップと起動

1. `local_streamdiffusion_td/local_streamdiffusion_td.toe`を開きます。
2. TouchDesignerとPythonを同じNVIDIA GPUで動かします。
3. カメラ映像をSpout Out TOPへ接続します。
4. Spout OutのSender Nameを`TD_Camera`にします。
5. 生成映像を受けるSpout In TOPで`AI_Output`を選択します。
6. Runを`1`にしてPythonと生成処理を起動します。

`local_streamdiffusion_launcher`が埋め込み済みです。
これを用いて、stream diffusionを起動します。初回は、絶対パスを指定してください。

Runは未起動時のPython起動に使用します。完全停止はOSC
`/streamdiffusion/system/stop 1`で行います。Stop後の次のRunでは、Python、
モデル、Spout senderを新しい単一プロセスとして起動します。

Python監視画面では次を確認できます。

- 入力 / 出力 / 推論FPS
- 推論時間とEnd-to-End時間
- GPU使用率、VRAM、GPU温度
- Spout入出力状態
- 現在適用中のOSCパラメータ
- 入出力プレビューとログ

OSC address一覧とTouchDesigner側の接続方法は
[TouchDesigner tools README](local_streamdiffusion_td/tools/README.md)を参照してください。

## オフライン本番運用

セットアップ、モデル取得、必要なTensorRTエンジンの作成が終わった後は、
インターネット接続なしで実行できます。`run.cmd`、`run_tensorrt.cmd`、
TouchDesignerランチャーはいずれもHugging Face / Transformers / Diffusersを
オフラインモードにして起動します。

本番前に、ネットワークを切った状態で以下を確認してください。

- 使用予定の全モデルが`local_streamdiffusion/models/`に存在する
- 使用予定の全TensorRT設定でエンジンが作成済み
- TouchDesignerからRun / Stop / パラメータ変更が動作する
- Spoutの`TD_Camera`と`AI_Output`が重複していない
- 30分以上の連続運転でVRAM使用量と温度が安定している

## すでにclone済みのPCを更新する

リポジトリのルートで実行します。

```powershell
git pull --ff-only origin main
cd .\local_streamdiffusion
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1 -SkipModels
.\setup_tensorrt.ps1
```

`models/`、`.venv/`、`.venv-trt/`、`engines/`、`config.json`はGit管理外なので、
通常の`git pull`では削除されません。GPU、モデル、解像度、step数、TensorRT環境を
変更した場合はエンジンを再ビルドしてください。

## Git管理に含めないもの

次のデータはPC固有または大容量のため、GitHubにはpushしません。

- `.venv/`、`.venv-trt/`
- `models/`
- `engines/`
- `.cache/`
- `benchmarks/`
- `local_streamdiffusion/config.json`
- TouchDesignerのBackup、Movie、Audio素材

`config.example.json`だけを共通テンプレートとして管理します。新しいPCでは
`config.json`がなくても既定値で起動し、設定受信後に自動保存されます。

## トラブルシューティング

### `Python 3.10 (64-bit) is required`

```powershell
py -0p
py -3.10 --version
```

Python 3.10が一覧にない場合は追加インストールします。既存のPython 3.11を
アンインストールする必要はありません。

### PowerShellでスクリプトを実行できない

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

または`.cmd`ファイルを使用してください。

### TensorRTエンジンが見つからない

現在のモデル、解像度、step数、GPUに一致するエンジンがありません。いったん
Backendを`Auto`または`xFormers`にし、設定を保存してから
`build_tensorrt_engine.cmd`を実行します。

### Spout映像が出ない

- TouchDesignerのSpout Out TOPがcookしているか確認
- Sender Nameが`TD_Camera`か確認
- Python出力とSpout In TOPが`AI_Output`か確認
- TouchDesignerとPythonを同じGPUへ割り当て
- 同名のPython推論プロセスが複数起動していないか確認

詳細なモデル設定、OSC address、FPS調整、時間安定化、TensorRT計測結果は
[Python backend README](local_streamdiffusion/README.md)を参照してください。
