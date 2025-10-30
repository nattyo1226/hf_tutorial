import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Load pretrained instances with an AutoClass

    本チュートリアルでは、`AutoClass` を用いて、モデルから目的のアーキテクチャをロードする手法を学びます。
    `pipeline()` が事前処理・推論・事後処理を一挙に行うモデル全体を提供していたのとは対照的に、`AutoClass` を用いることで、モデルを構成する各アーキテクチャを選択的にロードし、利用することができます。

    本チュートリアルは、[Hugging Face Tranformers チュートリアル](https://huggingface.co/docs/transformers/v4.57.0/ja/autoclass_tutorial) を元に、一部加筆・修正して作成しています。
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

    # %pip install pytesseract
    return


@app.cell
def _():
    import io
    import librosa
    from PIL import Image
    import requests
    from transformers import (
        AutoConfig,
        AutoFeatureExtractor,
        AutoImageProcessor,
        AutoModel,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
        AutoProcessor,
    )
    return (
        AutoConfig,
        AutoFeatureExtractor,
        AutoImageProcessor,
        AutoModel,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoProcessor,
        AutoTokenizer,
        Image,
        io,
        librosa,
        requests,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## `🤗 Tranformers` architecture

    一般的に、`🤗 Transformers` が提供するモデルは、次のような部品から構成されています。

    - processor: トーカナイザなどの、事前・事後処理を行うアーキテクチャ
    - model: モデル本体
    - head: model の出力をタスク固有の出力に変換するアーキテクチャ

    `pipeline()` ではこれらをまとめてロードしていましたが、`AutoClass` はこれらをより細かい単位でロードする仕組みが用意されています。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## AutoConfig

    model に関する情報は、`config.json` に保存されています。
    これらの情報を引き出すために、`AutoConfig` クラスを用います。
    `AutoConfig.from_pretrained("model")` とすることで、上述の `config.json` がダウンロードされ、その内容が格納された `xxxConfig` クラスのインスタンスが作成されます。
    """
    )
    return


@app.cell
def _(AutoConfig):
    # model: "openai/gpt2" (0.1B params)
    # ref: https://huggingface.co/openai-community/gpt2

    config = AutoConfig.from_pretrained("openai-community/gpt2")
    print(config)
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## AutoTokenizer

    **トーカナイザ**は、入力テキストに対してトークン化・テンソル化などの変換を行うアーキテクチャです。
    トーカナイザに関する情報は、トーカナイザを利用するような言語処理モデルの構成ファイルのうち、`tokenizer_config.json` に保存されています (以下で紹介する事前処理アーキテクチャについても、対応する config ファイルがモデルを構成するファイル群に含まれています) 。

    トーカナイザのロードには、`AutoTokenizer` クラスを用います。
    `AutoTokenizer` をはじめとする processor アーキテクチャ用のクラスには、`from_pretrained` メソッドが用意されています。
    `AutoTokenizer.from_pretrained("model name")` により、上述の config ファイルが読み込まれ、`xxxTokenizer` インスタンスが作成されます。
    """
    )
    return


@app.cell
def _(AutoTokenizer):
    # model: "google-bert/bert-base-uncased" (110M params)
    # ref: https://huggingface.co/google-bert/bert-base-uncased

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    print(tokenizer)

    sequence = "In a hole in the ground there lived a hobbit."
    print(tokenizer(sequence))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## AutoImageProcessor

    画像プロセッサは、入力画像にサイズ変更・正規化・テンソル化などの変換を行うアーキテクチャです。
    ここでは `AutoImageProcessor` クラスを用います。
    """
    )
    return


@app.cell
def _(AutoImageProcessor, Image, io, requests):
    # model: "google/vit-base-patch16-224" (86.6M params)
    # ref: https://huggingface.co/google/vit-base-patch16-224

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    print(image_processor)

    image1_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    image1 = Image.open(io.BytesIO(requests.get(image1_url).content))
    print(image_processor(image1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## AutoFeatureExtractor

    特徴抽出器は、入力の画像や動画に対して、一定の方法で特徴を抽出し、正規化・テンソル化などの変換を行うアーキテクチャです。
    ここでは、`AutoFeatureExtractor` クラスを用います。

    <audio src="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac" controls></audio>
    """
    )
    return


@app.cell
def _(AutoFeatureExtractor, io, librosa, requests):
    # model: "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition" (316M params)
    # ref: https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    )
    print(feature_extractor)

    target_sr = feature_extractor.sampling_rate
    speech_url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
    speech_bytes = io.BytesIO(requests.get(speech_url).content)
    speech, sr = librosa.load(speech_bytes, sr=None)
    if sr != target_sr:
        speech = librosa.resample(speech, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    print(feature_extractor(speech))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## AutoProcessor

    `Transformers` において、プロセッサとはマルチモーダルモデルの入力に対して前処理を行うアーキテクチャを指します。
    ここでは、`AutoProcessor` クラスを用います。

    <img src="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png" width="30%">
    """
    )
    return


@app.cell
def _(AutoProcessor, Image, io, requests):
    # model: "microsoft/layoutlmv2-base-uncased" (200M params)
    # ref: https://huggingface.co/microsoft/layoutlmv2-base-uncased

    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
    print(processor)

    image2_url = "https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png"
    image2 = Image.open(io.BytesIO(requests.get(image2_url).content)).convert("RGB")
    text = ["invoice", "number"]
    print(processor(images=[image2, image2], text=text, return_tensors="pt", padding=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## AutoModel

    model をロードするためには、`AutoModel` クラスを用います。
    `AutoModel` クラスには、`from_config` と `from_pretrained` の2種類のメソッドが用意されています。

    `from_config` メソッドは、`xxxConfig` インスタンスをもとに、`xxxModel` を組み立てます。
    この方法では、model を構成するパラメータはランダムに初期化されます。

    `from_pretrained` メソッドは、モデル名を受け取り、事前学習済みのパラメータをロードして、`xxxModel` を組み立てます。
    事前学習済み model を利用して、推論やファインチューニングを行うには、こちらを利用することになります。
    """
    )
    return


@app.cell
def _(AutoModel, config):
    # config = AutoConfig.from_pretrained("openai-community/gpt2")

    AutoModel.from_config(config)
    return


@app.cell
def _(AutoModel):
    AutoModel.from_pretrained("openai-community/gpt2")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## AutoModelFor

    model と同時に head もロードするためには、`AutoModelFor` クラス (`PyTorch` バックエンド) または `TFAutoModelFor` クラス (`tensorflow` バックエンド) を用います。
    ここでも、`AutoModel` と同様に、`from_config` メソッドと `from_pretrained` メソッドが用意されています。
    """
    )
    return


@app.cell
def _(AutoModelForSequenceClassification):
    # model: "distilbert/distilbert-base-uncased" (67M params)
    # ref: https://huggingface.co/distilbert/distilbert-base-uncased

    AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
    return


@app.cell
def _(AutoModelForTokenClassification):
    AutoModelForTokenClassification.from_pretrained("distilbert/distilbert-base-uncased")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
