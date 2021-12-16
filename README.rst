.. raw:: html

   <!--
    Copyright (c) 2021 flexfirm
   -->

Ensemble OCR
====================

画像から文字を読み取るOCRをアンサンブルで実行するディープラーニングモデルのPyTorch実装です。

-  `モデル概要 <#モデル概要>`__

   -  `ネットワーク構造 <#ネットワーク構造>`__
   -  `I/O <#io>`__
   -  `Data Augmentation <#data-augmentation>`__
   -  `Network Architecture <#network-architecture>`__
   -  `Learning Techniques <#learning-techniques>`__

-  `ディレクトリ構成 <#ディレクトリ構成>`__
-  `実行環境 <#実行環境>`__
-  `Dataset <#dataset>`__

   -  `データセットの準備 <#データセットの準備>`__
   -  `Exploratory Data Analysis <#exploratory-data-analysis>`__
   -  `設定JSONファイル <#設定jsonファイル>`__
   -  `TSVファイル <#tsvファイル>`__

-  `Training <#training>`__

   -  `config.iniの編集 <#configiniの編集>`__
   -  `Usage <#usage>`__
   -  `Tips <#tips>`__

      -  `Finetuning <#finetuning>`__
      -  `Scheduler <#scheduler>`__
      -  `Warm-up <#warm-up>`__
      -  `Checkpoint <#checkpoint>`__
      -  `Automatic Mixed Precision <#automatic-mixed-precision>`__

-  `Inference <#inference>`__

モデル概要
----------

ネットワーク構造
~~~~~~~~~~~~~~~~

.. code:: text

       image - CNN - pooling - RNN - FC - out

-  CNN: ResNet18
-  RNN: Bi-directional GRU 2-layers

I/O
~~~

-  Input: SJJの帳票から切り出した車台番号の画像
-  Output: 車台番号のテキスト

Data Augmentation
~~~~~~~~~~~~~~~~~

-  Resize to (320, 80)
-  Random Scaling
-  Random Rotation
-  Random Distortion
-  Random Crop to (256, 64)
-  Normalization from [0, 255] to [0.0, 1.0]
-  Mean Subtraction (channel wise)
-  Random Erasing

In details, see `transforms.py <./src/dataprocess/transforms.py>`__ and
`Dataset_info.json <./data/shatai_2.0.0/Dataset_info.json>`__

Network Architecture
~~~~~~~~~~~~~~~~~~~~

-  Pre-Activation ResNet > `Identity Mappings in Deep Residual Networks
   (arXiv) <https://arxiv.org/abs/1603.05027>`__
-  Bag of Tricks for Image Classification > `Bag of Tricks for Image
   Classification with Convolutional Neural Networks
   (arXiv) <https://arxiv.org/abs/1812.01187>`__

Learning Techniques
~~~~~~~~~~~~~~~~~~~

-  Learning Rate Warm-up > `Accurate, Large Minibatch SGD: Training
   ImageNet in 1 Hour (arXiv) <https://arxiv.org/abs/1706.02677>`__
-  Linear Scaling Learning Rate > `Don’t Decay the Learning Rate,
   Increase the Batch Size (arXiv) <https://arxiv.org/abs/1711.00489>`__


実行環境
--------

| Python 3.7.4
| PyTorch 1.4.0
| torchvision 0.5.0


| PyTorchの最新版はPyPIで配布されていないため、公式サイトを参照してインストールしてください。
| 参考：\ `PyTorch \| Get
  Started <https://pytorch.org/get-started/locally/>`__

その他の必要パッケージは以下のコマンドで一括インストールできます。

.. code:: bash

   $ pip install -r requirements.txt


Tutorial
~~~~~

.. code-block:: python

   import crnn_ocr

   guesses, confidences = crnn_ocr.recognize(image_path_list, batch_size)
   # guesses : list [str] : 推論結果
   # confidences : list [float] : 推論の確信度
