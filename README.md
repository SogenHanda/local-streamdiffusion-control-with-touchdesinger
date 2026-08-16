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

初版ではSpoutGLの`receiveImage` / `sendImage`を使用します。Spout自体はGPU共有ですが、このPython実装では推論前後にCPU画像バッファを経由します。まず安定したMVPとして動作を確認し、その後、必要に応じてCUDA/OpenGLテクスチャのゼロコピー化とTensorRT化を行います。

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

## 3. オフライン起動

モデル保存後はインターネット接続不要です。

```powershell
.\run.cmd
```

`run.cmd`はPowerShellの実行ポリシーに影響されません。`run.ps1`を使用する場合は、必要に応じて
`powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run.ps1`で起動してください。どちらもHugging Face、Transformers、Diffusersを明示的なオフラインモードにして起動します。

UIで設定を確認して「生成を開始」を押してください。起動直後はモデル読込に時間がかかります。最初のSpout入力を受信した時点で3フレーム分のウォームアップを行います。

## UI設定

- **モデル選択**: 導入済みモデルと追加候補を一覧表示。`models`直下のDiffusersモデルも「再検索」で検出
- **モデルパス**: ローカルモデルディレクトリ。カタログ外のパスも直接入力可能
- **動作プリセット**: Realtime（1-step + TinyVAE）、Balanced（2-step + TinyVAE）、Quality（2-step + Full VAE）
- **LCM-LoRA**: 高品質モデルを少ないステップで動かすローカルLoRA。初期値は`models/lcm-lora-sdv1-5`
- **Spout入力/出力**: TouchDesignerと一致させるSender Name
- **生成解像度**: 384〜1024を選択可能。リアルタイム用途は512または640を推奨
- **推論ステップ**: 1 stepは速度優先、2 stepsは画質優先
- **プロンプト**: 実行中も「プロンプトを即時更新」で変更可能
- **変換の強さ**: 100%に近いほど入力から大きく変化。実行中はモデルを再読込せず、StreamDiffusionのtimestepだけを再prepareして反映
- **入力フレーム保持**: 直前のカメラ入力を現在入力へ混ぜる割合。生成画像は推論入力へ戻さないため非再帰。実行中も即時反映
- **出力平滑化**: 新しい生成結果と前回出力を混ぜる割合。標準は15%。実行中も即時反映
- **生成特徴モーフ**: 直近の生成latentへ寄せる強さ。RGBを混ぜないため、残像よりも生成構造がもにゅっと移る効果になる。実行中も即時反映
- **特徴履歴**: 参照するlatentの枚数（2〜8）。3フレームが基準で、増やすほど変化は滑らかになるが反応は遅くなる
- **シーン変化リセット**: 入力の変化量がこの値を超えた場合、過去画像を使わず残像を防止。実行中も即時反映
- **Seed**: ノイズの初期値。実行中はモデルを再読込せず再prepareして反映
- **FPS上限**: 推論が十分速い場合の上限。実測FPSはGPUと設定で決まる。実行中も即時反映
- **TinyVAE**: VAEのエンコード/デコードを高速化。画質を優先する場合は無効化
- **LCM-LoRA**: DreamShaperへローカルLCM-LoRAを適用。SD-Turboを使う場合は無効化
- **オフライン固定**: ローカルにないモデルをネットから取得しない

プロンプト、変換強度、Seed、FPS上限、入力フレーム保持、出力平滑化、生成特徴モーフ、特徴履歴、シーン変化リセットは実行中に反映します。スライダー操作は180msでまとめて送るため、ドラッグ中にコマンドが過剰に溜まりません。モデル、Spout名、動作プリセット、解像度、推論ステップ、LCM-LoRA、TinyVAE、上下反転は停止後に再開して反映します。

## モデル比較

UIには次のモデルプロファイルがあります。

| モデル | 特徴 | LCM-LoRA | 推奨プリセット |
|---|---|---:|---|
| DreamShaper 8 | 汎用・製品・コンセプト表現 | 使用 | Balanced |
| Absolute Reality 1.81 | 人間の質感と作品表現のバランス | 使用 | Balanced |
| Realistic Vision 5.1 | 写真・素材感・実在感 | 使用 | Balanced |
| epiCRealism | 皮膚・布・素材の写真質感を優先 | 使用 | Quality |
| LCM DreamShaper v7 | 少ステップ用に直接蒸留 | 不要 | Realtime |
| SD-Turbo | 速度優先の比較基準 | 不要 | Realtime |

現在導入されているモデルにはUIで`【導入済み】`と表示されます。追加候補はオンライン環境で個別に取得できます。

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

- Spout入力FPS / 推論FPS / 出力FPS
- 1フレームの推論時間
- GPU使用率
- VRAM使用量
- GPU温度
- 入出力解像度
- 入力の動き量 / 実際に適用された入力フレーム保持率 / latentモーフ率
- 入出力プレビュー
- モデル読込、Spout接続、エラーのログ

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

Spout受信は推論とは別スレッドで動き、UIの`INPUT FPS`はPythonが実際に受信した新規Spoutフレーム数を表示します。推論中に届いた古いフレームはキューへ溜めず、常に最新の1枚だけを次の推論に使います。

それでもTouchDesignerが60 FPSなのに入力が低い場合は、次を確認してください。

1. `Spout Out TOP`のInfoで実際のcook rateが60になっているか
2. `Spout Out TOP`へ至るTOPチェーンが毎フレームcookされているか
3. TouchDesignerのPerform ModeとプロジェクトFPSが60になっているか
4. UIプレビューなど他の重いTOPを一時的に無効化して変化を見る

### 15〜30 FPSを優先したい

- 生成解像度: `512 × 512`（余裕があれば`640 × 640`）
- LCM推論ステップ: `1 step（高速）`
- `TinyVAE（高速）`: 有効
- FPS上限: `30`

`2 steps（高品質）`と通常VAEの組み合わせは画質優先です。RTX 4070 Tiで512 × 512を実測した時点では約5.6 FPSだったため、この設定のまま15〜30 FPSにはなりません。896/1024はVRAMに収まってもピクセル数に比例して計算量が増えるため、高FPS用ではなく高解像度出力用です。

### 生成結果が前の絵に引っ張られすぎる

1. 「入力フレーム保持」を35%から20〜25%へ下げる
2. 「出力平滑化」を15%から5〜10%へ下げる
3. UIの「時間履歴をリセット」を押す

人物や物体が大きく移動する用途ではRGB保持率を低めにします。固定カメラで従来の入力保持だけを使う場合は20〜35%が目安ですが、残像を避ける用途では下のlatentモーフを優先してください。

### 残像ではなく、もにゅっとモーフさせたい

UIの「残像を抑えるモーフ推奨値」を押すか、次を基準に調整します。

- 入力フレーム保持: 0〜5%
- 出力平滑化: 0〜5%
- 生成特徴モーフ: 35〜45%
- 特徴履歴: 3フレーム（より緩やかにしたいときは4）
- シーン変化リセット: 30〜40%

`LATENT MORPH`には入力の動きに応じて実際に適用された割合が表示されます。履歴は再帰的に蓄積せず、指定枚数を超えたlatentを破棄します。大きなシーン変化ではStreamDiffusion内部バッファも同時にリセットします。

### UIだけ停止して見える

モデル読込または1フレームの推論中は、停止要求がその処理の終了後に反映されます。ログ欄で状態を確認してください。

## テスト

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
