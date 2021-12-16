# Vision models - PyTorch

Implementation of vision models, ResNet, gMLP, CvT in PyTorch. 

- [Directory structure](#directory-structure)
- [Environment](#environment)
- [Dataset](#dataset)
  - [画像ファイルの配置とCSVファイル](#画像ファイルの配置とcsvファイル)
  - [設定JSONファイル](#設定jsonファイル)
  - [Usage - Dataset](#usage---dataset)
- [Training](#training)
  - [Preparation](#preparation)
  - [ベースモデルの学習](#ベースモデルの学習)
  - [スタッキングモデルの学習](#スタッキングモデルの学習)
- [Inference](#inference)

## Directory structure

```text
/
├── README.md
├── analysis/                       <- 精度検証に用いるデータ・notebooks
├── data/                           <- データセットの設定JSON
├── models/                         <- 訓練済みモデルファイル・学習チェックポイント
├── logs/                           <- TensorBoard用の学習ログ
├── notebooks/                      <- Jupyter notebooks
｜  └── dataset/                    <- データセットの作成・改変用スクリプト
├── results/                        <- 推論結果ファイルの出力先
├── requirements.txt                <- パッケージインストールに必要なパッケージ
├── devRequirements.txt             <- 開発に必要なパッケージ
└── vision_models/
    ├── dataprocess/                <- DataLoader, and so on.
    ├── functions/                  <- Evaluation, and so on.
    ├── models/                     <- DNN models.
    ｜  ├── cvt/                    <- Convolutional Transformer
    ｜  ├── gmlp/                   <- gMLP
    ｜  ├── resnet/                 <- ResNet
    ｜  └── model.py                <- LightningModule
    ├── args.py                     <- Execution arguments.
    └── inference.py                <- Inference script
```

## Environment

- Python 3.8.5  
- PyTorch 1.10.0  
- PyTorch Lightning 1.5.5

PyTorchの最新版はPyPIで配布されていないため、公式サイトの導入手順を参照してください。  
[PyTorch | Get Started](https://pytorch.org/get-started/locally/)

また、ロギングにTensorBoardを使用する場合はTensorFlowまたはTensorBoardをインストールしてください。

```bash
$ pip install tensorflow
```

その他の必要パッケージは以下のコマンドで一括インストールできます。

```bash
$ pip install -r devRequirements.txt
```

## Dataset

### 画像ファイルの配置とCSVファイル

画像ファイルパスとラベルをそれぞれx列、y列として記述したCSVファイルを作成し、次のように配置してください：

```text
/
├── train.csv
├── val.csv
├── test.csv
└── images/
```

CSVに記述するパスはCSVの親ディレクトリを基準とした相対パスで記述してください。  
画像ファイルはCSVのパスが正しい限り、どのように配置しても構いません。

### 設定JSONファイル

<details>
<summary>データセットの設定はJSONに記述します。例えば次のような形式です。</summary>

```json
{
    "dataset_name": "numeric classification dataset",
    "version": "200116_Siogy第1回定期保守データ",
    "dataset_information": {
        "root_path": "\\ADP1\AI\9.dataset\ai_ocr_numeric_v2_1_0",
        "train": "train_ai_ocr_numeric_v4_0_1.csv",
        "val": "val_ai_ocr_numeric_v4_0_1.csv",
        "test": "val_ai_ocr_numeric_v4_0_1.csv",
        "color_mode": "RGB",
        "img_size": [32, 32],
        "labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "×", "-", "・", "/", "字", "―"]
    },
    "transform_parameters": {
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "random_affine": {
            "degrees": [-0.20, 0.20],
            "translate": [0.10, 0.10],
            "scale": [0.9, 1.1]
        },
        "random_erase": {
            "p_erase": 0.5,
            "scale_range": [0.01, 0.03],
            "aspect_ratio": [0.03, 3.3],
            "fill_value": "random"
        }
    }
}
```

</details>

<details>
<summary>各項目の意味は次の通りです。</summary>

|プロパティ|サブプロパティ|データ型|説明|
|---|---|---|---|
|dataset_name|||データセットの名前(任意)
|version|||データセットのバージョン(任意)
|dataset_information|||データセットの情報
||root_path|`str`|データセットのルートパス<br>空文字列でこのJSONの親ディレクトリを参照する
||train<br>val<br>test|`str`|`root_path`から各CSVファイルへの相対パス
||color_mode|`str`|画像のカラーモードに応じて`"RGB"` or `"Gray"`を指定
||img_size|`list [int]`|モデルへ入力する画像サイズ (H, W)
||labels|`list [str]`|クラスindexと対応するラベルを設定(任意)
|transform_parameters|||画像前処理のパラメータ
|normalize|||Normalization (Mean subtraction)
||mean<br>std|`list [float]`|全データのチャンネルごとのピクセル値の平均値と標準偏差<br>RGB画像ではRGBの順に指定し、グレースケール画像では長さ1のリストで指定する
|random_affine|||アフィン変換をランダム実行
||degrees|`list [float]`|回転の範囲
||translate|`list [int]`|画像のランダムシフトの範囲 (`[x軸, y軸]`で画像サイズに対するシフト幅の最大値を指定)
||scale|`list [float]`|拡大縮小の範囲
|random_erase|||Random Erasing<br>画像の一部を特定の値、またはランダム値で置き換える
||p_erase|`float`|Random Erasingの実行確率
||scale_range|`list [float]`|消去矩形の大きさの範囲を元画像に対する比率で指定
||aspect_ratio|`list [float]`|アスペクト比(H/W)の範囲
||fill_value|`int`<br>or `list [int]`<br>or `str`|消去矩形を塗りつぶすピクセル値<br>`"random"`指定の時はランダム値をとる

</details>



### Usage - Dataset

[`notebooks/test_dataloader.ipynb`](notebooks/test_dataloader.ipynb)も参照。

```python
# pseudo code
json_path = Path('../data/dataset_info.json')
args = Args(json_path=json_path)
train_dataset = dataprocess.build_dataset('train', args)
train_dataloader = dataprocess.build_dataloader('train', args)
```

---

## Training

### Preparation

事前に[`vision_models/args.py`](vision_models/args.py)のデフォルトハイパーパラメータを確認してください。  
これらの値は次のようにして実行時に変更可能です。

<details>

```python
arg_params = {
    'epochs': 200,
    'batch_size': 512,
    'nb_classes': 17,
    'eval_interval': 5,
    # ResNet parameters
    'resnet_layers': 18,
    'use_se_module': False,
    # optimizer
    'optimizer': 'adaberief',
    'lr': 1e-3,
    'weight_decay': 1e-4,
    # scheduler
    'scheduler': 'cosine',
    'warmup': True,
    'warmup_epoch': 5,
    
    'device': 'cuda',
    'log_dir': '../logs',  # logのルートディレクトリ
    'project': 'model-ensemble',  # logファイルのディレクトリ
    'model': 'resnet',
    'version': 'exclude-maxpool',
    'json_path': '../data/dataset_info_1.json'
}

args = Args(**arg_params)
```

</details>

### ベースモデルの学習

それぞれ次のnotebookで訓練してください。

- [`notebooks/train_resnet.ipynb`](notebooks/train_resnet.ipynb)
- [`notebooks/train_cvt.ipynb`](notebooks/train_cvt.ipynb)
- [`notebooks/train_gmlp.ipynb`](notebooks/train_gmlp.ipynb)

### スタッキングモデルの学習

上記のベースモデルを2つ以上訓練し、訓練済み重みファイル(`.pth`)を保存します。  
[`notebooks/train_stacking.ipynb`](notebooks/train_stacking.ipynb)を実行し、pthファイルで複数の訓練済みモデルをロードしてスタッキングモデルを訓練できます。

## Inference

スタッキングモデルの推論は[`notebooks/inference_stacking.ipynb`](notebooks/inference_stacking.ipynb)で実行できます。

各ベースモデルに関しては、上記訓練用notebooksの末尾に推論用のスクリプトがあるので参考にしてください。
