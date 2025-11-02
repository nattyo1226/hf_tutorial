import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Preprocess

    テキストや音声、画像をモデルに渡して推論・訓練を行うためには、事前にそれらをモデルが期待する形式に変換しておく必要があります。
    本チュートリアルでは、`Transformers` ライブラリが提供する、データの事前処理の手法について学びます。

    本チュートリアルは、[Hugging Face Transformers チュートリアル](https://huggingface.co/docs/transformers/v4.57.1/ja/preprocessing) を元に、一部加筆・修正して作成しています。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Dependencies

    このチュートリアルコードをすべて実行するためには、明示的に `import` するライブラリの他に、以下のソフトウェアが必要です。

    - [`tesseract`](https://github.com/tesseract-ocr/tesseract) (および、その Python ラッパー: `pytesseract`): 画像処理 (ocr)
    - `torch` ライブラリ or `tensorflow` ライブラリ: バックエンド
        - 本チュートリアルでは `torch` を用いるコードしか紹介しません
    """
    )
    return


@app.cell
def _():
    # run this cell if you are working in google colab

    # %pip install pytesseract torchcodec
    return


@app.cell
def _():
    from datasets import load_dataset, Audio
    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision.transforms import (
        ColorJitter,
        Compose,
        RandomResizedCrop,
    )
    from transformers import (
        AutoConfig,
        AutoFeatureExtractor,
        AutoImageProcessor,
        AutoProcessor,
        AutoTokenizer,
    )
    return (
        Audio,
        AutoConfig,
        AutoFeatureExtractor,
        AutoImageProcessor,
        AutoProcessor,
        AutoTokenizer,
        ColorJitter,
        Compose,
        RandomResizedCrop,
        load_dataset,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Natural Language Processing

    自然言語処理のタスクにおいて、テキストの事前処理に使用する主なアーキテクチャは**トーカナイザ**です。
    トーカナイザは、テキストを一定のルールのもとで**トークン**に分割します。
    トークンは単語や文字、あるいは単語を構成する部分文字列で構成されます。
    個々のトークンに識別番号を振ることでテキストが数列に変換され、これにより、機械学習モデルが文字列を数理的に取り扱えるようになります。

    ここでも、`from_pretrained()` メソッドを使用します。
    """
    )
    return


@app.cell
def _(AutoTokenizer):
    # model: google-bert/bert-base-cased (110M params)
    # ref: https://huggingface.co/google-bert/bert-base-cased

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    return (tokenizer,)


@app.cell
def _(tokenizer):
    encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
    print(encoded_input)
    return (encoded_input,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ここで、トーカナイザの出力について補足しておきます。

    - `input_ids`: 文中の各トークンに対応するインデックス
    - `token_type_ids`: 複数の文が入力された場合に、それらを区別するために付与される id の列
    - `attention_mask`: attention アーキテクチャがトークンを受け取る必要があるかを示す bool の列

    `input_ids` をデコードすることで元の入力が得られます。
    ここでわかるように、トーカナイザは文章に自動的に特別なトークン (`CLS`, `SEP`) を付与します。
    """
    )
    return


@app.cell
def _(encoded_input, tokenizer):
    tokenizer.decode(encoded_input["input_ids"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""複数の文章の前処理を行うこともできます。""")
    return


@app.cell
def _(tokenizer):
    batch_sentences = [
        "But what about second breakfast?",
        "Don't think he knows about second breakfast, Pip.",
        "What about elevensies?",
    ]
    encoded_inputs1 = tokenizer(batch_sentences)
    print(encoded_inputs1)
    return (batch_sentences,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Padding

    テキストは常に同じ長さ (同じトークン数) とは限りませんが、推論モデルはある特定の長さの入力しか受け付けることができません。
    そこで、トーカナイザはテキストをトークン化しつつ、その長さを揃えることが期待されます。

    このための戦略の1つがパディングです。
    `padding=True` を指定することで、入力バッチ中の最長のテキストに合わせて、短いテキストに**パディングトークン**が追加されます。
    """
    )
    return


@app.cell
def _(batch_sentences, tokenizer):
    encoded_inputs2 = tokenizer(batch_sentences, padding=True)
    print(encoded_inputs2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Truncation

    入力テキストの長さが、モデルが期待する入力次元を超えてしまう場合があります。
    `truncation=True` を指定することで、モデルが受け入れる最大の長さにトークン列を切り詰めます。
    """
    )
    return


@app.cell
def _(batch_sentences, tokenizer):
    encoded_inputs3 = tokenizer(batch_sentences, padding=True, truncation=True)
    print(encoded_inputs3)
    return


@app.cell
def _(mo):
    mo.md(r"""ちなみに、今回利用している "google-bert/bert-base-cased" モデルが浮き入れる最大トークン数は 512 であるため、`batch_sentences` に含まれている入力テキスト程度のトークン数では Truncation は有効に効いてきません。""")
    return


@app.cell
def _(AutoConfig):
    bert_config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
    print(bert_config.max_position_embeddings)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Build tensors

    `return_tensors="pt"` (`"tf"`) を指定することで、出力を `PyTorch` (`TensorFlow`) のテンソル形式に変換します。
    """
    )
    return


@app.cell
def _(batch_sentences, tokenizer):
    encoded_inputs4 = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
    print(encoded_inputs4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Audio

    音声処理タスクにおいて、音声データの事前処理に使用する主なアーキテクチャは**特徴抽出器**です。特徴抽出器は生の音声データから特徴を抽出し、それらをテンソルに変換します。

    まず、入力データセットをロードします。
    """
    )
    return


@app.cell
def _(load_dataset):
    dataset_audio = load_dataset("PolyAI/minds14", name="en-US", split="train")
    dataset_audio.features
    return (dataset_audio,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    `dataset_audio.features` によると、サンプリングレートは 8 kHz であるようです。
    今回は Wav2Vec2 モデルへの入力を想定しますが、このモデルはサンプリングレート 16 kHz のデータで事前学習されているので、`dataset_audio` を 16 kHz でリサンプルしましょう。
    """
    )
    return


@app.cell
def _(Audio, dataset_audio):
    dataset_audio2 = dataset_audio.cast_column("audio", Audio(sampling_rate=16000))
    dataset_audio2.features
    return (dataset_audio2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""次に、特徴抽出器を用いて入力データを正規化します。""")
    return


@app.cell
def _(AutoFeatureExtractor):
    # model: facebook/wav2vec2-base (95M params)
    # ref: https://huggingface.co/facebook/wav2vec2-base

    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    return (feature_extractor,)


@app.cell
def _():
    # audio_input1 = [dataset_audio2[0]["audio"]["array"]]
    # print(feature_extractor(audio_input1))
    # audio_input2 = [dataset_audio2[1]["audio"]["array"]]
    # print(feature_extractor(audio_input2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    テキストデータと同様に、データセットに含まれる音声データの長さがすべて等しいとは限らず (`shape` メンバを参照) 、また、モデルが期待する入力の長さには限りがあります。
    そこで、パディングとトランケーションを行います。
    特徴抽出器では、最大サンプル長を制御するために `max_length=<number>` を指定します。
    """
    )
    return


@app.cell
def _(feature_extractor):
    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            padding=True,
            max_length=100000,
            truncation=True,
        )
        return inputs
    return (preprocess_function,)


@app.cell
def _(dataset_audio2, preprocess_function):
    processed_dataset_audio = preprocess_function(dataset_audio2[:5])
    print(processed_dataset_audio["input_values"][0].shape)
    print(processed_dataset_audio["input_values"][1].shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Computer Vision

    画像処理タスクにおいて、画像データの事前処理に使用する主なアーキテクチャは**画像プロセッサ**です。
    画像プロセッサは元の画像データに対してリサイズ・正規化・チャネル
    補正などの処理を行い、それらをテンソルに変換します。

    例によって、データセットと画像プロセッサを読み込みます。
    """
    )
    return


@app.cell
def _(load_dataset):
    dataset_cv = load_dataset("food101", split="train[:100]")
    dataset_cv[0]["image"]
    return (dataset_cv,)


@app.cell
def _(AutoImageProcessor):
    # model: google/vit-base-patch16-224 (86.6M params)
    # ref: https://huggingface.co/google/vit-base-patch16-224

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    return (image_processor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    続いて、データセットに含まれる画像データを、所定の方法で処理していきます。
    ここでは、`torchvision` の `transforms` モジュールを使用します。
    ここで行う処理は以下のとおりです。

    - `RandomResizedCrop`: モデルが期待する画像サイズ (`image_size`) に合わせて、画像をランダムに切り抜く。
    - `ColorJitter`: 画像の色調や明るさをランダムに変化させる。
        - `brightness=0.5`: 明るさを $\pm 50 \%$ の範囲でランダムに変化させる。
        - `hue=0.5`: 色相値を $\pm 0.5$ の範囲でランダムに変化させる。
    """
    )
    return


@app.cell
def _(ColorJitter, Compose, RandomResizedCrop, image_processor):
    image_size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )

    transforms_core = Compose([
        RandomResizedCrop(image_size),
        ColorJitter(brightness=0.5, hue=0.5),
    ])
    return (transforms_core,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""`transforms_core` をデータセットの各画像に適用し、画像プロセッサで処理します。""")
    return


@app.cell
def _(image_processor, transforms_core):
    def transforms(examples):
        images = [transforms_core(img.convert("RGB")) for img in examples["image"]]
        examples["pixel_values"] = image_processor(images, do_resize=False, return_tensors="pt")["pixel_values"]
        return examples
    return (transforms,)


@app.cell
def _(dataset_cv, transforms):
    dataset_cv.set_transform(transforms)
    dataset_cv[0].keys()
    return


@app.cell
def _(dataset_cv, plt):
    img = dataset_cv[0]["pixel_values"]
    plt.imshow(img.permute(1, 2, 0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Padding

    データセットに含まれる画像のサイズが異なる場合には、`DataImageProcessor.pad()` によってパディングを施します。
    """
    )
    return


@app.cell
def _(image_processor):
    def collate_fn(batch):
        pixel_values = [item["pixel_values"] for item in batch]
        encoding = image_processor.pad(pixel_values, return_tensors="pt")
        labels = [item["labels"] for item in batch]
        batch = {}
        batch["pixel_values"] = encoding["pixel_values"]
        batch["pixel_mask"] = encoding["pixel_mask"]
        batch["labels"] = labels
        return batch
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Multi Modal

    マルチモーダルタスクにおいて、データの事前処理に使用する主なアーキテクチャは**プロセッサ**です。 プロセッサはトーカナイザや特徴抽出器などの複数の事前処理アーキテクチャを結合します。

    まずはデータセットを読み込みます。
    [オリジナルのチュートリアル](https://huggingface.co/docs/transformers/v4.57.1/ja/preprocessing) では `lj_speech` というデータセットが読み込まれていますが、このデータセットは最新の `datasets` (version 4.3.0) ではサポートされていないので、別のデータセットをダウンロードします。
    """
    )
    return


@app.cell
def _(load_dataset):
    librispeech1 = load_dataset("hf-internal-testing/librispeech_asr_demo", split="validation")
    librispeech1.features
    return (librispeech1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""今回興味があるのは `audio` と `text` だけなので、それ以外のメンバを削除してしまいます。""")
    return


@app.cell
def _(librispeech1):
    librispeech2 = librispeech1.map(remove_columns=["file", "id", "chapter_id", "speaker_id"])
    print(librispeech2.features)
    print(librispeech2[0]["audio"]["sampling_rate"])
    print(librispeech2[0]["text"])
    return (librispeech2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    `librispeech_asr_demo` データセットはサンプリングレートが 16 kHz でモデルの事前学習データセットのサンプリングレートと一致しているので、リサンプリングの必要はありません。
    安心してモデルを読み込みましょう。
    """
    )
    return


@app.cell
def _(AutoProcessor):
    # model: facebook/wav2vec2-base-960h (95M params)
    # ref: https://huggingface.co/facebook/wav2vec2-base-960h

    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    return (processor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""`processor` に `audio` と `text` を指定して、事前処理を行います。""")
    return


@app.cell
def _(processor):
    def prepare_dataset(example):
        audio = example["audio"]

        example.update(processor(audio=audio["array"], text=example["text"], sampling_rate=16000))

        return example
    return (prepare_dataset,)


@app.cell
def _(librispeech2, prepare_dataset):
    prepare_dataset(librispeech2[0])
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
