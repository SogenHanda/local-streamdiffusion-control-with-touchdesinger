# Ergonomics Local Diffusion Bridge

TouchDesignerからSpoutで受け取った映像を、ローカルのStreamDiffusionで連続img2img処理し、SpoutでTouchDesignerへ戻すWindows用アプリです。標準構成はDreamShaper 8 + LCM-LoRAの1-stepリアルタイム推論です。直近2〜8枚の生成latent（VAE変換前の生成特徴）を動き量に応じてモーフさせることで、RGB画像の重ね合わせで生じる残像や白い輪郭を抑えながら、フレームごとの急激な絵柄変化を抑えます。APIやクラウドサービスは使用しません。

## 構成

```text
TouchDesigner
  Movie Device In TOP
    → Resolution TOP (512 × 512)
    → Spout Out TOP [TD_Camera]
          ↓
Python UI / StreamDiffusion / RTX GPU
          ↓
    Spout sender [AI_Output]
          ↓
TouchDesigner
  Spout In TOP
    → 後処理・表示・録画
```

SpoutGLの`receiveImage` / `sendImage`を使用するため、推論前後にはCPU画像バッファを経由します。受信は最新フレームのみを保持し、既定30fpsでサンプリングすることで、TouchDesignerが60fpsでも推論に使わないフレームのGPU readbackとPIL変換を抑えます。推論部は従来のxFormersに加え、モデル・LCM-LoRA・解像度・step数・GPUごとに固定したTensorRT FP16 UNetを選択できます。

## 動作対象

- Windows 10 / 11 64-bit
- NVIDIA RTX GPU（このプロジェクトではRTX 4070 Ti 12GBを確認済み）
- NVIDIAドライバー
- Python 3.10 64-bit
- TouchDesigner 2023または2025

## 1. 初回セットアップ（インターネット接続あり）

PowerShellでこのディレクトリへ移動し、次を実行します。

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

以下を行います。

1. `.venv`へPython環境を作成
2. CUDA 12.1版PyTorch、xFormers、StreamDiffusion、SpoutGLを導入
3. `models/dreamshaper-8`、`models/lcm-lora-sdv1-5`、`models/taesd`へモデルを保存
4. CUDAと必要ライブラリを検査

モデルを後で取得する場合は次のようにします。

```powershell
.\setup.ps1 -SkipModels
.\download_models.ps1
```

モデル利用条件は導入前に必ず確認してください。

- [DreamShaper 8 model card](https://huggingface.co/Lykon/dreamshaper-8)
- [Absolute Reality 1.81 model card](https://huggingface.co/Lykon/absolute-reality-1.81)
- [epiCRealism model card](https://huggingface.co/emilianJR/epiCRealism)
- [LCM-LoRA SD 1.5 model card](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)
- [TAESD](https://huggingface.co/madebyollin/taesd)

### TensorRT高速化セットアップ（推奨・初回のみ）

TensorRTは既存の`.venv`を変更せず、`.venv-trt`へ分離して導入します。オンライン環境で実行してください。PyTorch、cuDNN、TensorRTと1モデル分のengineを含めて約12GB使用するため、作業前に15GB以上の空きを確保してください。

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_tensorrt.ps1
```

次に、UIで使用するモデル・解像度・step数・LCM-LoRAを選び、「現在設定をビルド」を押します。コマンドから行う場合は次です。初回ビルドは10〜30分程度かかることがあります。

```powershell
.\build_tensorrt_engine.cmd
```

ビルド後はTensorRT専用環境で起動します。

```powershell
.\run_tensorrt.cmd
```

推論バックエンドを`Auto（TensorRT優先）`にすると、現在の条件に一致するエンジンがあればTensorRTを使い、なければxFormersへ安全に戻ります。`TensorRT FP16`を明示選択した場合は、エンジンがなければ開始せずビルド方法を表示します。

エンジンは次の条件が変わると再ビルドが必要です。

- モデルまたはLCM-LoRAの変更
- 生成解像度の変更
- 1-step / 2-stepの変更
- GPUの変更（RTX 4070 Tiで作ったエンジンをRTX 4090 Laptopへコピーしない）
- TensorRT/CUDAランタイムの変更

プロンプト、Seed、変換強度、時間安定化、TinyVAEのON/OFFではUNetエンジンの再ビルドは不要です。512 × 512 / 2-stepのUNet engineはモデルごとに約1.8GBです。ビルド成功後は約3.2GBの中間ONNXを自動削除します。`engines/`はGPU固有の大容量生成物なのでGit管理しません。別PCではリポジトリをpull後、`setup_tensorrt.ps1`と`build_tensorrt_engine.cmd`をそのPC上で実行してください。

## 2. TouchDesigner側

### 入力

1. `Movie Device In TOP`を配置してカメラを選択
2. `Resolution TOP`を接続し、出力を`512 × 512`に設定
3. `Spout Out TOP`を接続
4. Sender Nameを`TD_Camera`に設定

画角を歪ませたくない場合は、`Resolution TOP`の前で正方形に中央クロップしてください。Python側でも中央クロップしますが、TouchDesigner側で構図を確認できる方が扱いやすくなります。

### 出力

1. `Spout In TOP`を配置
2. Sender Nameから`AI_Output`を選択
3. 必要な`Level TOP`、`Blur TOP`、合成、表示先へ接続

Spout名をUIで変更した場合は、TouchDesigner側も同じ名前にしてください。映像が上下逆の場合はUIの「入力を上下反転」または「出力を上下反転」を切り替え、いったん停止して再開します。

### TouchDesignerからOSC制御

Python UIは起動時に`127.0.0.1:13001/UDP`でOSC受信を開始します。TouchDesignerのOSC Out DATはNetwork Addressを`127.0.0.1`、Portを`13001`にしてください。追加Pythonパッケージは不要で、通常のOSC MessageとOSC Bundleの両方を受信できます。

実行中にモデルを読み直さず反映する値:

| OSC address | 型 | 範囲・内容 |
|---|---|---|
| `/ergonomics/live/prompt` | string | 空でないプロンプト |
| `/ergonomics/live/strength` | float | 0.0〜1.0。0はほぼ入力、1は最大変換 |
| `/ergonomics/live/seed` | int | -1〜2147483647 |
| `/ergonomics/live/target_fps` | float | 0.1〜240 |
| `/ergonomics/live/input_feedback` | float | 0.0〜0.8 |
| `/ergonomics/live/output_smoothing` | float | 0.0〜1.0 |
| `/ergonomics/live/latent_morph` | float | 0.0〜1.0 |
| `/ergonomics/live/history_frames` | int | 2〜8 |
| `/ergonomics/live/motion_threshold` | float | 0.0〜1.0 |

再初期化が必要な設定:

| OSC address | 型 | 範囲・内容 |
|---|---|---|
| `/ergonomics/config/model_index` | int | 下記モデル表のindex |
| `/ergonomics/config/backend_index` | int | 0=Auto、1=TensorRT、2=xFormers、3=PyTorch |
| `/ergonomics/config/width` | int | 64以上かつ8の倍数 |
| `/ergonomics/config/height` | int | 64以上かつ8の倍数 |
| `/ergonomics/config/performance_index` | int | 0=Realtime、1=Balanced、2=Quality |
| `/ergonomics/config/spout_input` | string | Spout入力Sender Name |
| `/ergonomics/config/spout_output` | string | Spout出力Sender Name |
| `/ergonomics/config/spout_sample_fps` | float | 1〜240 |

`model_index`は固定で、`0=DreamShaper 8`、`1=Absolute Reality 1.81`、`2=Realistic Vision 5.1`、`3=epiCRealism`、`4=LCM DreamShaper v7`、`5=SD-Turbo`です。モデルに応じたLCM-LoRAのON/OFF、Performanceに応じたstep数とTinyVAEはPython側で決定します。上下反転は入力・出力ともOFF、オフラインモードはON、TensorRT CUDA GraphはOFFに固定しています。

設定値は連続して届くことを想定し、最後の受信から350ms後にまとめて反映・保存します。推論中なら自動的に安全に停止して新設定で再開し、停止中なら設定だけを保存します。そのためTouchDesigner側のApply操作では各設定を再送するだけでよく、`/ergonomics/config/apply`を送る必要はありません。互換用として同addressへ`1`を送れば待たずに即時確定できます。

操作trigger:

| OSC address | 値 | 動作 |
|---|---:|---|
| `/ergonomics/system/start` | 1 | 生成開始 |
| `/ergonomics/system/stop` | 1 | 生成停止 |
| `/ergonomics/temporal/reset` | 1 | 時間安定化履歴を手動リセット |

triggerの`0`は無視するため、TouchDesignerのボタンが`1 → 0`と変化しても操作は1回だけ実行されます。受信値はPython UIにも同期され、範囲外・未対応address・型違いは推論へ渡さずログへ理由を表示します。

OSC Out CHOPが全チャンネルを毎フレーム送る構成でも、Python側は1回のUI更新につきaddressごとの最新値だけを使用し、前回と同じ設定値は推論へ再送しません。Start / Stop / Reset / Applyも`1`の立ち上がりだけを処理します。これにより60fpsのOSC入力で監視UIや推論ワーカーが飽和することを防ぎます。TouchDesigner側でも可能なら、パラメータは値が変化したときだけ送る構成を推奨します。

TouchDesignerのRun値との接続と、`.toe`起動時にPythonを自動起動するコードは[`touchdesigner/README.md`](touchdesigner/README.md)にあります。使用するText DAT本体は[`touchdesigner/local_inference_launcher.py`](touchdesigner/local_inference_launcher.py)です。

## 3. オフライン起動

モデル保存後はインターネット接続不要です。

```powershell
.\run.cmd
```

`run.cmd`はPowerShellの実行ポリシーに影響されません。`run.ps1`を使用する場合は、必要に応じて
`powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run.ps1`で起動してください。どちらもHugging Face、Transformers、Diffusersを明示的なオフラインモードにして起動します。

TensorRTを使用する場合は`run.cmd`ではなく`run_tensorrt.cmd`を使用してください。通常の`run.cmd`はTensorRTを含まない既存`.venv`で起動するため、Auto設定ではxFormersを使用します。

Python UIは監視専用です。生成開始・停止とパラメータ変更はTouchDesignerからOSCで行います。起動直後のモデル読込には時間がかかり、最初のSpout入力を受信した時点で3フレーム分のウォームアップを行います。

## Python監視UI

Python側には編集欄・スライダー・設定保存・開始停止ボタンを表示しません。表示内容は次の通りです。

- OSC待受状態、総受信数、最後に受信した時刻・address・値
- 全OSC addressの最新受信値
- 現在適用中のモデル、バックエンド、Performance、解像度、Spout I/O
- 現在適用中の変換強度、Seed、FPS、時間安定化設定
- OSCから受け取った現在のプロンプト（読取専用）
- 入出力映像プレビュー
- Spout送信元推定FPS、Python入力FPS、推論FPS、出力FPS
- 推論時間、End-to-End時間、各処理区間の時間
- GPU使用率、VRAM使用量、GPU温度
- 入出力解像度、動き量、実適用された入力保持率・latentモーフ率
- モデル読込、設定変更、Spout接続、OSCエラー、推論エラーのログ

アプリ起動時は`config.json`の最後の設定を表示し、その後はTouchDesignerから受信した値で監視表示と推論設定を同期します。

## モデル比較

`model_index`では次のモデルプロファイルを選択できます。

| モデル | 特徴 | LCM-LoRA | 推奨プリセット |
|---|---|---:|---|
| DreamShaper 8 | 汎用・製品・コンセプト表現 | 使用 | Balanced |
| Absolute Reality 1.81 | 人間の質感と作品表現のバランス | 使用 | Balanced |
| Realistic Vision 5.1 | 写真・素材感・実在感 | 使用 | Balanced |
| epiCRealism | 皮膚・布・素材の写真質感を優先 | 使用 | Quality |
| LCM DreamShaper v7 | 少ステップ用に直接蒸留 | 不要 | Realtime |
| SD-Turbo | 速度優先の比較基準 | 不要 | Realtime |

追加候補はオンライン環境で個別に取得できます。

```powershell
.\download_models.ps1 --preset absolute-reality-1.81
.\download_models.ps1 --preset epicrealism
.\download_models.ps1 --preset lcm-dreamshaper-v7
.\download_models.ps1 --preset realistic-vision-v5.1
```

一覧だけ確認する場合:

```powershell
.\.venv\Scripts\python.exe .\download_models.py --list-presets
```

モデルの変更は画風や1-step時の破綻傾向を改善できますが、同じSD 1.5 UNetなので同一解像度・同一stepで速度差は小さめです。まず`Balanced — 2-step + TinyVAE`を基準に比較し、FPS不足の場合は`LCM DreamShaper v7`の1-step、次に512から448/384への解像度低下を試します。2-stepを保ったまま大幅に高速化する次の段階はTensorRT化です。SDXL系モデルは現行のSD 1.5用LCM-LoRAとTAESDに互換性がないため、このモデル一覧には含めていません。

## 監視情報

- Spout送信元推定FPS / Python取得FPS / 推論FPS / 出力FPS
- 1フレームの推論時間 / Spout受信から送信完了までのEnd-to-End時間
- Spout受信・送信、前処理、VAE Encode、UNet、VAE Decode、後処理の区間別時間
- 実際に使用中のバックエンド
- GPU使用率
- VRAM使用量
- GPU温度
- 入出力解像度
- 入力の動き量 / 実際に適用された入力フレーム保持率 / latentモーフ率
- 入出力プレビュー
- モデル読込、Spout接続、エラーのログ

## 実測と比較方法

RTX 4070 Ti 12GB、epiCRealism、512 × 512、2-step、TinyVAEで推論単体を60フレーム測定した結果です。Spout/TouchDesignerを含む実測値はTOP構成とGPU競合で変わります。

| バックエンド | 平均 | p95 | 推論FPS | UNet平均 |
|---|---:|---:|---:|---:|
| xFormers | 66.73 ms | 72.02 ms | 14.99 | 47.91 ms |
| TensorRT FP16 | 48.27 ms | 49.50 ms | 20.72 | 27.69 ms |

TensorRTは同条件で平均フレーム時間を約28%短縮し、推論FPSを約38%向上しました。固定入力で保存したxFormers/TensorRT出力はMAE 1.58/255、PSNR 39.7 dBで、目視でも同じ椅子構造を維持しています。StreamDiffusion 0.1.1のTensorRTラッパーは毎フレーム入出力バッファを再確保するため、CUDA Graphを使うと古いバッファアドレスを再生してノイズ化します。本実装では画質を優先してCUDA Graphを無効化しています。

任意のPC・設定で再計測できます。結果はGit管理外の`benchmarks/`へJSON保存されます。

```powershell
.\.venv-trt\Scripts\python.exe .\benchmark_pipeline.py --backend xformers --frames 60
.\.venv-trt\Scripts\python.exe .\benchmark_pipeline.py --backend tensorrt --frames 60
```

## トラブルシューティング

### `Spout入力待ち`のまま

- TouchDesignerの`Spout Out TOP`がcookしているか確認
- Sender Nameが`TD_Camera`になっているか確認
- TouchDesignerとPythonアプリを同じGPUで実行
- ノートPCでは両方をWindowsの「高パフォーマンスGPU」に割り当て

### モデルがない

オンライン環境で次を実行します。

```powershell
.\download_models.ps1
```

### CUDAを認識しない

```powershell
.\.venv\Scripts\python.exe .\verify_install.py
nvidia-smi
```

### `RuntimeError: Numpy is not available`

PyTorch 2.1.0に対してNumPy 2.xが導入された場合に発生します。次を実行してNumPy 1.26.4へ戻してください。

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall numpy==1.26.4
```

`requirements.txt`では再発防止のためNumPy 1.26.4に固定しています。

### VRAM不足

1. 生成解像度を512 × 512へ戻す
2. TinyVAEを有効化
3. 他のGPUアプリを閉じる
4. TouchDesignerの不要な高解像度TOPを解放

### INPUT FPSがTouchDesignerのFPSより低い

Spout受信は推論とは別スレッドで動き、UIの`INPUT FPS`はPythonが実際にCPUへ取得した新規Spoutフレーム数です。既定の`Spout取得`が30なので、TouchDesignerが60fpsでもINPUT FPSは最大約30になります。これは推論に使用しないフレームのGPU readbackを避ける意図した動作です。`SOURCE FPS`はSpoutフレーム番号の進み方から送信元FPSを推定します。推論中に届いた古いフレームはキューへ溜めず、常に最新の1枚だけを次の推論に使います。

それでもTouchDesignerが60 FPSなのに入力が低い場合は、次を確認してください。

1. `Spout Out TOP`のInfoで実際のcook rateが60になっているか
2. `Spout Out TOP`へ至るTOPチェーンが毎フレームcookされているか
3. TouchDesignerのPerform ModeとプロジェクトFPSが60になっているか
4. UIプレビューなど他の重いTOPを一時的に無効化して変化を見る

送信元60fpsを検証する必要がある場合だけ`Spout取得`を60へ上げられますが、生成が15〜20fpsなら通常は30の方がCPU負荷と遅延を抑えられます。

### 15〜30 FPSを優先したい

- 生成解像度: `512 × 512`（余裕があれば`640 × 640`）
- LCM推論ステップ: `1 step（高速）`
- `TinyVAE（高速）`: 有効
- FPS上限: `30`

`2 steps（高品質）`と通常VAEの組み合わせは画質優先です。896/1024はVRAMに収まってもピクセル数に比例して計算量が増えるため、高FPS用ではなく高解像度出力用です。写実モデルの品質を維持する本番基準は、`512 × 512 / 2-step / TinyVAE / TensorRT FP16 / Spout取得30`です。

### 生成結果が前の絵に引っ張られすぎる

1. 「入力フレーム保持」を35%から20〜25%へ下げる
2. 「出力平滑化」を50%から25〜35%へ下げる
3. UIの「時間履歴をリセット」を押す

人物や物体が大きく移動する用途ではRGB保持率を低めにします。固定カメラで従来の入力保持だけを使う場合は20〜35%が目安ですが、残像を避ける用途では下のlatentモーフを優先してください。

### 残像ではなく、もにゅっとモーフさせたい

UIの「残像を抑えるモーフ推奨値」を押すか、次を基準に調整します。

- 入力フレーム保持: 0%
- 出力平滑化: 35〜50%
- 生成特徴モーフ: 55〜65%
- 特徴履歴: 6〜8フレーム
- 動き追従しきい値: 70〜80%

`LATENT MORPH`には入力の動きに応じて実際に適用された割合が表示されます。履歴は再帰的に蓄積せず、指定枚数を超えたlatentを破棄します。大きな変化でも履歴やStreamDiffusion内部バッファを自動的にゼロへ戻しません。

変換強度またはSeedを実行中に変更した場合は、2-step推論に必要な内部latentを直近のカメラ入力から非表示で再充填します。これにより、設定変更直後にゼロlatent由来の茶色・灰色の単色フレームが出る問題を防ぎます。

### UIだけ停止して見える

モデル読込または1フレームの推論中は、停止要求がその処理の終了後に反映されます。ログ欄で状態を確認してください。

## テスト

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
