import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Load pretrained instances with an AutoClass

    本チュートリアルでは、`AutoClass.from_pretrained()` メソッドを用いて、訓練済みモデルから目的のアーキテクチャをロードする手法を学びます。
    `pipeline()` が事前処理・推論・事後処理を一挙に行うモデル全体を提供していたのとは対照的に、`AutoClass.from_pretrained()` はモデルを構成する各アーキテクチャを選択的にロードし、別のタスクのために利用することができます。

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

    - [`tesseract`](https://github.com/tesseract-ocr/tesseract) (および、その Python ラッパー: `pytesseract`): 動画処理
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
        AutoFeatureExtractor,
        AutoImageProcessor,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
        AutoProcessor,
    )
    return (
        AutoFeatureExtractor,
        AutoImageProcessor,
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
    ## AutoTokenizer

    Tokinizer は、入力テキストに対してトークン化・テンソル化などの変換を行うアーキテクチャです。
    ここでは、`AutoTokenizer` クラスを用います。

    以下のサンプルコードではいくつか出力が表示されますが、その意味については事前処理についてのチュートリアルに譲り、ここでは詳しい説明を省きます。
    """
    )
    return


@app.cell
def _(AutoTokenizer):
    # model: "google-bert/bert-base-uncased" (110M params)
    # ref: https://huggingface.co/google-bert/bert-base-uncased

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    sequence = "In a hole in the ground there lived a hobbit."

    tokenizer(sequence)
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
    image1_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    image1 = Image.open(io.BytesIO(requests.get(image1_url).content))

    image_processor(image1)
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
    target_sr = feature_extractor.sampling_rate
    speech_url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
    speech_bytes = io.BytesIO(requests.get(speech_url).content)
    speech, sr = librosa.load(speech_bytes, sr=None)
    if sr != target_sr:
        speech = librosa.resample(speech, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    feature_extractor(speech)
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
    image2_url = "https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png"
    image2 = Image.open(io.BytesIO(requests.get(image2_url).content)).convert("RGB")
    text = ["invoice", "number"]

    processor(images=[image2, image2], text=text, return_tensors="pt", padding=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## AutoModel

    特定のタスクに対して訓練済みモデルを `torch.Tensor` (`tf.Tensor`) 形式でロードするためには、`AutoModelFor` クラス (`TFAutoModelFor` クラス) を用います。
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
