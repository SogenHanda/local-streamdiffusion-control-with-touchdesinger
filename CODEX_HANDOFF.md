# Codex引き継ぎ書 — Local StreamDiffusion + TouchDesigner

最終更新: 2026-09-18

## 1. このプロジェクトの目的

TouchDesignerのカメラ映像をSpoutでローカルPythonへ渡し、Windows上のNVIDIA RTX
GPUでStreamDiffusion系のリアルタイムimg2imgを実行して、生成映像をSpoutで
TouchDesignerへ戻すシステムです。

本番操作はTouchDesignerへ集約しています。Pythonは推論、Spout入出力、OSC受信、
GPU監視を担当し、Pythonウィンドウは読取専用の監視画面です。クラウドAPIは使わず、
モデル取得と初回セットアップ後はオフラインで運用します。

## 2. GitHubとブランチ

- Repository: `https://github.com/SogenHanda/local-streamdiffusion-control-with-touchdesinger`
- Active branch: `main`
- Remote `origin`: 上記の新しい汎用リポジトリ
- Remote `legacy-origin`: 旧プロジェクトの履歴保持用。通常はpushしない

作業開始時は必ず以下を確認してください。

```powershell
git status --short
git remote -v
git pull --ff-only origin main
```

## 3. 現在の有効な構成

```text
repository root/
├─ local_streamdiffusion/       Python / CUDA / TensorRT / Spout / OSC
├─ local_streamdiffusion_td/    TouchDesigner .toeと補助Text DATソース
├─ README.md                    新しいPCへのセットアップ手順
├─ CODEX_HANDOFF.md             この引き継ぎ書
└─ AGENTS.md                    Codex用の作業ルール
```

アクティブな製品コードは上の2フォルダだけです。元PCの`archive_*`、
`assets_source_video_edit`、`reference_tox_analysis`はGit管理外の過去資料で、
通常は変更しません。

### 映像と制御の流れ

```text
TouchDesigner camera
  → Spout Out: TD_Camera
  → Python StreamDiffusion
  → Spout Out: AI_Output
  → TouchDesigner Spout In

TouchDesigner → OSC 127.0.0.1:13001 → Python control
Python monitor → OSC 127.0.0.1:9001 → TouchDesigner dashboard
```

## 4. 重要な設計判断

### TouchDesignerをメイン操作画面にする

- Python UIには編集欄やStart/Stopボタンを置かない。
- Python UIはFPS、推論時間、GPU、VRAM、Spout、OSC値、入出力プレビュー、ログを表示。
- モデル、Backend、Performance、解像度、プロンプト、時間安定化はTDから送る。

### Start / Stopのライフサイクル

- Run=1: Pythonが未起動なら起動し、`--autostart`で生成開始。
- Run=0: 何もしない。誤停止や毎フレーム再起動を防ぐための仕様。
- `/streamdiffusion/system/stop 1`: 推論ワーカーだけでなくPythonアプリ、モデル、
  Spout senderを完全終了。
- 次のRun=1: 新しい単一プロセスをクリーン起動。
- ランチャーはOSC 13001の待受を検出して二重起動を防ぐ。

### 時間方向の安定化

- RGBの強いフィードバックだけに依存せず、生成latent履歴をモーフする。
- `input_feedback`はカメラ入力履歴を使い、生成出力を再帰的に入力しない。
- `output_smoothing`は小さな変化だけを平滑化し、大きな変化では残像を抑える。
- `latent_morph`と短い時間幅のlatent履歴でパラパラした生成変化を減らす。
- 現在の`history_frames`は30fps換算の時間幅（2=33ms、8=233ms）。
  `latent_morph.py`でraw latentの有限時間窓を平均し、局所的な入力の変化で残像を抑える。
  2-stepの入力ガイドは1フレーム遅らせる。時間幅内のraw履歴は最大64件で制限する。
- RGBのOutput Smoothingは100%でも履歴を最大50%に制限し、現在フレームを残す。
- 自動のscene-cutリセットは無効。`motion_threshold`は徐々に追従量を変える。
- Strength / Seed変更時は内部latentを非表示で再充填し、茶色や灰色の単色フレームを
  抑止する。

### TensorRT

- xFormers、PyTorch、TensorRT FP16を選択可能。
- Autoは一致するTensorRTエンジンがあれば使用し、なければxFormersへfallback。
- TensorRT明示指定では一致するエンジンがなければ開始せず、誤条件を隠さない。
- Engine keyはGPU、モデル、解像度、step、LCM-LoRA、runtimeに依存。
- CUDA Graphは既定OFF。`config.json`の`tensorrt_cuda_graph=true`で、新しい固定バッファ
  UNetとTinyVAEのgraphを有効化できる。OSC設定と監視UIはこの値を維持する。
  StreamDiffusion標準ラッパーは毎フレーム再確保でノイズ化するため使わない。
  `tensorrt_unet.py`はshape/device/dtype変更で再確保・再captureし、入力を毎回更新する。
  TinyVAE graphはstatelessなencode/decodeだけをcaptureし、時間履歴やnoiseを含めない。
- TensorRT engineとONNX中間生成物はGit管理しない。

## 5. OSCプロトコル

名前空間はすべて`/streamdiffusion`です。旧クライアント固有namespaceへ
戻さないでください。

### リアルタイム反映

モデルを再読込せずに更新します。

| Address | Type | Range / meaning |
|---|---|---|
| `/streamdiffusion/live/prompt` | string | 空でないprompt |
| `/streamdiffusion/live/strength` | float | 0.0〜1.0。0はほぼ入力、1は最大変換 |
| `/streamdiffusion/live/seed` | int | -1〜2147483647 |
| `/streamdiffusion/live/target_fps` | float | 0.1〜240 |
| `/streamdiffusion/live/input_feedback` | float | 0.0〜0.8 |
| `/streamdiffusion/live/output_smoothing` | float | 0.0〜1.0 |
| `/streamdiffusion/live/latent_morph` | float | 0.0〜1.0 |
| `/streamdiffusion/live/history_frames` | int | 2〜8 |
| `/streamdiffusion/live/motion_threshold` | float | 0.0〜1.0 |

### 再初期化を伴う設定

| Address | Type | Meaning |
|---|---|---|
| `/streamdiffusion/config/model_index` | int | model tableのindex |
| `/streamdiffusion/config/backend_index` | int | 0 Auto / 1 TensorRT / 2 xFormers / 3 PyTorch |
| `/streamdiffusion/config/performance_index` | int | 0 Realtime / 1 Balanced / 2 Quality |
| `/streamdiffusion/config/width` | int | 64以上、8の倍数 |
| `/streamdiffusion/config/height` | int | 64以上、8の倍数 |
| `/streamdiffusion/config/spout_input` | string | 入力Spout名 |
| `/streamdiffusion/config/spout_output` | string | 出力Spout名 |
| `/streamdiffusion/config/spout_sample_fps` | float | 1〜240 |
| `/streamdiffusion/config/apply` | trigger | debounceを待たず即時反映 |

設定は最後の受信から350ms後にまとめて保存・反映します。推論中は安全に再初期化し、
停止中は保存だけ行います。同じ値の連続受信とtriggerのfalling edgeは無視します。

### Trigger

| Address | Meaning |
|---|---|
| `/streamdiffusion/system/start` | 生成開始 |
| `/streamdiffusion/system/stop` | アプリ完全終了 |
| `/streamdiffusion/system/shutdown` | TD終了時の完全終了 |
| `/streamdiffusion/temporal/reset` | 時間安定化履歴の手動reset |

monitor addressの完全な一覧は`local_streamdiffusion/README.md`を参照してください。

## 6. Indexの固定対応

### Model index

| Index | Model |
|---:|---|
| 0 | DreamShaper 8 + LCM-LoRA |
| 1 | Absolute Reality 1.81 + LCM-LoRA |
| 2 | Realistic Vision 5.1 + LCM-LoRA |
| 3 | epiCRealism + LCM-LoRA |
| 4 | LCM DreamShaper v7 |
| 5 | SD-Turbo |

### Performance index

| Index | Preset | Details |
|---:|---|---|
| 0 | Realtime | 1-step + TinyVAE |
| 1 | Balanced | 2-step + TinyVAE |
| 2 | Quality | 2-step + Full VAE |

この対応はTouchDesigner側と共有されるプロトコルです。並び順を変える場合は、
Python、TD UI、README、testsを同時に更新してください。

## 7. 主要ファイル

### Python

| File | Responsibility |
|---|---|
| `local_streamdiffusion/app.py` | CLI entry point、monitor表示、autostart |
| `streamdiffusion_bridge/config.py` | AppConfig、validation、local config保存 |
| `streamdiffusion_bridge/model_catalog.py` | model/performance indexの固定表 |
| `streamdiffusion_bridge/osc_control.py` | OSC decode、validation、monitor bundle |
| `streamdiffusion_bridge/runtime.py` | Spout受信thread、推論worker、最新frame保持 |
| `streamdiffusion_bridge/engine.py` | StreamDiffusion、live更新、時間安定化 |
| `streamdiffusion_bridge/tensorrt_backend.py` | engine key、inspect/build/load |
| `streamdiffusion_bridge/tensorrt_unet.py` | 固定バッファUNet、CUDA stream同期、任意のgraph |
| `streamdiffusion_bridge/cuda_graph.py` | TinyVAE graphのcapture/replay/解放 |
| `streamdiffusion_bridge/image_processing.py` | FP16正規化を維持したGPU上の8bit画像変換 |
| `streamdiffusion_bridge/timing.py` | Windows高精度のフレーム待機 |
| `streamdiffusion_bridge/latent_morph.py` | 時間幅での生成特徴モーフ、局所入力変化による履歴抑制 |
| `streamdiffusion_bridge/spout_transport.py` | SpoutGL receive/send wrapper |
| `streamdiffusion_bridge/ui.py` | 読取専用monitorとOSC lifecycle |
| `download_models.py` | offline用model preset download |
| `build_tensorrt_engine.py` | 現在のconfigに一致するengine build |

### TouchDesigner

| File | Responsibility |
|---|---|
| `local_streamdiffusion_td/local_streamdiffusion_td.toe` | 現在の本体project |
| `tools/local_streamdiffusion_launcher.py` | sibling backend検出、起動、完全停止 |
| `tools/build_control_ui.py` | control UI生成補助 |
| `tools/build_monitor_ui.py` | OSC monitor dashboard生成補助 |
| `tools/README.md` | TD callbackとlauncher接続方法 |

`.toe`にはlauncherとmonitor builderが埋め込まれています。外部`.py`は編集・比較用の
source of truthです。両方を変更した場合は`.toe`内のText DATも同期してください。

重要: 最新の`.toe`に埋め込まれた`local_streamdiffusion_launcher`には、直近で
作業したPC用の絶対`APP_DIR`が保存されています。別PCで最初に開いたときは、次の
どちらかを行ってから`.toe`を保存してください。

1. 埋め込みText DATの内容を`tools/local_streamdiffusion_launcher.py`で置き換える
   （推奨。兄弟フォルダを自動検出する）。
2. 埋め込みText DATの`APP_DIR`を、そのPCの`local_streamdiffusion`絶対パスへ変更する。

古いPCの`C:\Users\...`が残ったままRunしないでください。

## 8. Git管理外のローカルデータ

以下は意図的にcloneされません。

- `local_streamdiffusion/.venv/`
- `local_streamdiffusion/.venv-trt/`
- `local_streamdiffusion/models/`
- `local_streamdiffusion/engines/`
- `local_streamdiffusion/.cache/`
- `local_streamdiffusion/benchmarks/`
- `local_streamdiffusion/config.json`
- `local_streamdiffusion_td/Backup/`
- TDのMovie / Audio / Chan / Geo / Image素材

新しいPCではREADMEのセットアップを実行し、モデルとTensorRT engineをそのPCで
作成してください。`config.example.json`が共通テンプレートです。

## 9. 検証済み状態

元PCで確認済み:

- Windows / NVIDIA GeForce RTX 4070 Ti 12GB
- Python 3.10 virtual environments
- PyTorch 2.1.0+cu121
- StreamDiffusion 0.1.1
- TensorRT 9.0.1
- 52 unit tests pass
- TensorRT verification pass
- OSC名前空間は`.toe`再展開でも`/streamdiffusion`へ統一されていることを確認済み
- 最新の`.toe`には別PC用の絶対`APP_DIR`があるため、target PCで更新が必要

参考実測（epiCRealism、512×512、2-step、TinyVAE、RTX 4070 Ti）:

| Backend | Mean frame | Inference FPS | UNet mean |
|---|---:|---:|---:|
| xFormers | 66.73 ms | 14.99 | 47.91 ms |
| TensorRT FP16 | 48.27 ms | 20.72 | 27.69 ms |

TouchDesignerを含む実FPSはTOP構成、GPU競合、camera/Spout条件で変化します。

## 10. テストと確認コマンド

```powershell
cd .\local_streamdiffusion
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe .\verify_install.py
.\.venv-trt\Scripts\python.exe .\verify_tensorrt.py
```

TensorRT benchmark:

```powershell
.\.venv-trt\Scripts\python.exe .\benchmark_pipeline.py --backend xformers --frames 60
.\.venv-trt\Scripts\python.exe .\benchmark_pipeline.py --backend tensorrt --frames 60
```

変更後は少なくとも`git diff --check`、unit tests、`git status --short`を確認します。

### RTX 4090 PCでの高速化検証（2026-09-18）

- デスクトップRTX 4090 24GB / Driver 581.08 / Python 3.10.11 / TensorRT 9.0.1.post11.dev4。
- Realistic Vision 5.1・512×512・TinyVAEの前処理を含む固定入力計測:
  Balancedは28.63→33.37fps、1-stepは30.40→37.24fps。
- 各200フレームで入力反転、Strength/Seed、prompt、history/smoothing/morph変更を比較。
  従来TensorRT実装との最大画素差はどちらも0/255。
- 29fps指定で約21fpsになるWindows待機問題を修正。実出力が入力上限に近づいたら、
  TD側の`spout_sample_fps`を60へ上げて計測する。画質設定は維持する。
- CPUスレッド数変更とcuDNN自動探索は、このPCで明確な効果がなく採用していない。
- CUDA Graphの起動直後はcaptureがある。ウォームアップ後の値で比較する。
- `frame_fps`は前処理込み。旧benchmarkの`fps`と直接混ぜて比較しない。
- 他のGPU・モデル・解像度でのgraph品質比較と30分耐久は別途必要。

#### 原因の切り分けと最終確認

- 受信だけの比較で、通常待機は21.18fps、Windowsタイマー精度を一時的に上げると
  28.83fps、元へ戻すと21.17fpsになった。製品実装はグローバルなタイマー設定変更を
  避け、受信thread専用の高精度waitable timerを使用する。
- 毎フレームのTensorRT buffer再確保とbinding設定を除去。固定アドレスを維持して
  UNetをcaptureし、TinyVAEのencode/decodeも個別にcaptureする。
  呼出元と専用CUDA streamの依存関係を明示し、入力を毎回更新する。
- 画像変換はGPU上で8bit化してからCPUへ転送する。FP16での正規化とfloat32での
  丸め順序を維持し、有限FP16値の全パターンで従来変換との一致をテストした。
- 品質比較の基準はcommit `18853a0`のTensorRT実装。固定入力を12回warmup後、
  200フレーム実行。30フレームごとに入力を左右反転し、40フレーム目にStrength/Seed、
  80フレーム目にprompt、120/160フレーム目に時間安定化設定を変更した。
  Balancedと1-stepそれぞれについて全フレームの画素を比較した。
- 実映像では修正前の約21fpsから、入力取得上限60fps設定後は約30fpsを確認。
  この間に利用者が入力映像や時間履歴設定を変更しているため、この実映像の値は
  厳密な同条件比較ではない。同条件での改善率は上記固定入力比較を使う。
- 最終のBalanced設定でも120フレーム再測定し、`frame_fps=33.05`を確認。
  `fps=37.07`は前処理を除く別指標であり、実出力FPSではない。
- unit testsは60件成功。CUDAのbuffer再利用・入力更新・shape変更時の再capture、
  timer解放、OSCと監視UIによるgraph設定保持を含む。TensorRT verifyも成功。
- OSC Stopでcontrollerが終了すること、通常launcherで起動すること、Run=0で
  停止しないこと、二重起動防止、graph設定と入力取得上限60の保持を確認した。
- このPCの`config.json`はgraph有効。Git管理される既定値はfalseなので、別PCでは
  停止中に明示的に有効化して品質と安定性を検証する。engineの再buildは不要。
- 生ログ、benchmark JSON、個人の映像、ローカル設定はGit管理しない。

## 11. 既知の注意点

### モーフ更新の検証

- Realistic Vision / TensorRT CUDA Graph / 512×512 / Balancedで実Spout入力を120枚取得し、
  同じ入力とprompt変更を用いて各180フレーム比較。処理全体は32.87→33.91ms
  （30.43→29.49fps）。滑らかさを増やす処理の追加コストはこの測定で約1ms。
- 入力を再生した区間の平均フレーム間画素変化は0.830→0.621/255。
  この値だけで画質や残像の優劣は判定できない。短いクリップでの確認であり、
  高速動作・遮蔽・他モデル・長時間の主観品質は未検証。
- 69 testsとTensorRT verifyが成功。段階的・単調な遷移、時間幅後の収束、
  静止画細部の保持、局所的な動きでの残像抑制、2-stepの入力位置合わせを含む。
  最終テスト初回に既存GPU画像変換テストでPythonが1度異常終了（0xC0000409）。
  単独テストと全68件の再実行では再現せず、原因は未特定。実推論比較は完走。
- TensorRTエンジンの再buildは不要。現行のモデル・解像度・ステップ数を維持する。

- Python 3.11ではなく3.10を使う。既存Pythonは削除せずside-by-sideでよい。
- PyTorch 2.1.0ではNumPy 2.xを使わない。`numpy==1.26.4`に固定済み。
- `SOURCE FPS=0`はSpout送信元frame IDを取得できない場合があり、必ずしも入力停止ではない。
- INPUT FPSはPythonがCPUへ取得したframe数。TDが60fpsでもsample FPS設定により低くなる。
- TensorRT engineがなくてもAutoならxFormersへfallbackする。TensorRT明示指定はfallbackしない。
- `.toe`をTouchDesignerで開いたまま外部から置換すると、古いメモリ状態のSaveで戻る。
  `.toe`更新時はTDを閉じるか、別名保存とbackupを使う。
- 最新の埋め込みlauncherはPC固有`APP_DIR`を持つ。新PCでは外部`tools`版へ同期するか
  絶対パスを更新する。
- `config.json`は個人のprompt、model、Spout設定を含むためcommitしない。
- 大容量modelやengineをGitへ追加しない。

## 12. 次のPCで最初に行うこと

1. ルート`README.md`に従いPython 3.10、標準環境、モデルを導入。
2. `.toe`内のlauncherを`tools`版へ同期するか、新PCの`APP_DIR`へ変更。
3. 本番GPU上でTensorRT環境を導入。
4. 最初はBackend Auto/xFormersでTDからmodel・resolution・performanceを保存。
5. アプリをStopして、その設定のTensorRT engineを本番PCでbuild。
6. unit tests（この変更時点で60件）とTensorRT verifyを実行。
7. Spout I/O、OSC 13001/9001、Run/Stop再起動を確認。
8. RTX 4090 Laptopでモデル別FPSと画質を比較。
9. 本番前に30分以上の耐久テストを行う。

耐久テストはまだ最終実施していません。次の開発優先事項は、RTX 4090 Laptop上で
本番候補モデルごとのengineを作り、画質、FPS、VRAM、温度、長時間安定性を記録する
ことです。
