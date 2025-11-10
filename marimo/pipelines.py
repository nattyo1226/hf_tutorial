import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pipelines for inference

    本チュートリアルでは、`pipeline()` を用いて、訓練済みモデルによって様々な推論タスクを実行する手法を学びます。
    ここで紹介するモデルのうちいくつかは非常に多くのパラメータをもつため、ロードのために多くのメモリやストレージを必要とします。
    ローカルで実行する際は注意してください。

    本チュートリアルは、[Hugging Face Transformers チュートリアル](https://huggingface.co/docs/transformers/v4.57.1/ja/pipeline_tutorial) を元に、一部加筆・修正して作成しています。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dependencies

    このチュートリアルコードをすべて実行するためには、明示的に `import` するライブラリの他に、以下のソフトウェアが必要です。

    - [`ffmpeg`](https://www.ffmpeg.org/): 音声処理
    - [`tesseract`](https://github.com/tesseract-ocr/tesseract) (および、その Python ラッパー: `pytesseract`): 画像処理 (ocr)
    - `accelerate` ライブラリ: モデルの自動配置
    - `bitsandbytes` ライブラリ: モデルの量子化
        - linux, windows のみサポート
    - `pillow` ライブラリ: 画像処理
    - `torchcodec` ライブラリ: 音声処理

    もし自分の環境にインストールされていない場合には、事前にインストールしておいてください。\
    なお、`ffmpeg` と `tesseract` に関しては、macOSであれば [`Homebrew`](https://formulae.brew.sh) から簡単にインストールできるようです (動作未確認) 。
    """)
    return


@app.cell
def _():
    # run this cell if you are working in google colab

    # %pip install bitsandbytes torchcodec pytesseract
    return


@app.cell
def _():
    # import dependencies

    from datasets import load_dataset
    import torch
    from transformers import pipeline
    from transformers.pipelines.pt_utils import KeyDataset
    return KeyDataset, load_dataset, pipeline, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pipeline usage

    `pipeline(task="task")` により、推論タスク `"task"` を行うためのデフォルトのモデルが提供されます。
    提供される構造体は、入力に対して事前処理・推論・事後処理をワンライナーで実行します。

    <audio src="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac" controls></audio>
    """)
    return


@app.cell
def _(pipeline):
    # default model: "facebook/wav2vec2-base-960h" (94.4M params)
    # ref: (https://huggingface.co/facebook/wav2vec2-base-960h)

    pipe_asr1 = pipeline(task="automatic-speech-recognition")
    pipe_asr1("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
    return (pipe_asr1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    具体的な推論モデルを指定するには、`pipeline(model="model")` とします。
    モデルの一覧は [`Hub`](https://huggingface.co/models) から確認できます。
    """)
    return


@app.cell
def _(pipeline):
    # prepare generator with model name
    # superior model: "openai/whisper-large" (1.54B params)
    # this may be too heavy for cpus ...

    pipe_asr2 = pipeline(model="openai/whisper-large")
    pipe_asr2("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    複数の入力を `list` で受け取ることもできます。

    <audio src="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac" controls></audio>

    <audio src="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac" controls></audio>
    """)
    return


@app.cell
def _(pipe_asr1):
    pipe_asr1([
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parameter

    `pipeline()` はタスク固有・非固有の多くのパラメータをサポートしています。
    一般的には、このパラメータはどこでも指定できます。

    ```python
    pipe = pipeline(..., my_parameter=1)
    out = pipe(...)                  # `my_parameter=1` is used here
    out = pipe(..., my_parameter=2)  # `my_parameter=2` is used here
    out = pipe(...)                  # `my_parameter=1` is used here
    ```

    以下で、特によく用いられるパラメータを紹介します。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Device

    `device=n` を指定すると、モデルが指定したデバイスのメモリに配置されます。
    具体的な use case は以下の通りです。

    - `device=-1`: CPU
    - `device=n` (non-negative integer): on GPU with id `n`
        - id はマシンを構成する各 GPU に自動的に割り振られています
        - NVIDIA GPU であれば `torch.cuda.get_device_name(n)` で id `n` に対応する GPU デバイス名が取得できます

    なお、特に `device` の値を指定しなくても、GPU を使用するように自動的にデバイスが決定されるようです。
    筆者の環境 (M4 MacBook Air) では、Apple GPU (mps) が自動的に選択されました。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Batch Size

    `batch_size=n` を指定することで、バッチサイズ `n` で推論することができます。
    ただし、バッチ処理によって実行速度の向上が必ずしも期待できるわけではなく、いくつかのケースではかなり遅くなることが確認されているようです。
    なお、バッチ処理を行ったとしても、得られる結果はバッチ処理を行わない場合と一致します。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task specific parameters

    すべてのタスクにおいて、タスク固有のパラメータが提供されています。
    例えば、`transformers.AutomaticSpeechRecognitionPipeline.call()` メソッドには、適当な単位で推論結果を区切ってタイムスタンプと同時に出力する `return_timestamps` パラメータがあります。
    """)
    return


@app.cell
def _(pipeline):
    # model: "facebook/wav2vec2-large-960h-lv60-self" (317M params)
    # ref: https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self

    pipe_asr3 = pipeline(model="facebook/wav2vec2-large-960h-lv60-self", return_timestamps="word")
    pipe_asr3("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using pipeline in a dataset

    `pipeline()` は大規模なデータセット上で推論を実行することもできます。
    """)
    return


@app.cell
def _(pipeline):
    # model: "openai-community/gpt2" (137M params)
    # ref: https://huggingface.co/openai-community/gpt2

    def data():
        for i in range(10):
            yield f"My example {i}"

    pipe_tg1 = pipeline(model="openai-community/gpt2", device=0)
    for out_tg1 in pipe_tg1(data()):
        print(out_tg1[0]["generated_text"][:250], "...")
        print("---\n")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `🤗 Datasets` からデータセットをロードして繰り返し反復させることもできます。
    """)
    return


@app.cell
def _(KeyDataset, load_dataset, pipeline):
    pipe_asr4 = pipeline(model="hf-internal-testing/tiny-random-wav2vec2", device=0)
    dataset_asr1 = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:10]")

    for out_asr1 in pipe_asr4(KeyDataset(dataset_asr1, "audio")):
        print(out_asr1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using pipelines for a webserver

    このセクションは飛ばします。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vision pipeline

    画像処理タスクでの使用例は以下の通りです。
    ここでは、写真に写っているオブジェクトを分類する推論タスクを実行しています。

    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg" width="30%">
    """)
    return


@app.cell
def _(pipeline):
    # model: "google/vit-base-patch16-224" (86.6M params)
    # ref: https://huggingface.co/google/vit-base-patch16-224

    pipe_vc1 = pipeline(model="google/vit-base-patch16-224")
    preds_vc1 = pipe_vc1(
        inputs="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    )
    preds_vc1 = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds_vc1]
    preds_vc1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Text pipeline

    テキスト処理タスクでの使用例は以下の通りです。
    ここでは、テキストのコンテンツの性格を分類するタスクを実行しています。
    """)
    return


@app.cell
def _(pipeline):
    # model: "facebook/bart-large-mnli" (407M params)
    # ref: https://huggingface.co/facebook/bart-large-mnli

    pipe_tc1 = pipeline(model="facebook/bart-large-mnli")
    pipe_tc1(
        "I have a problem with my iphone that needs to be resolved asap!!",
        candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multimodal pipeline

    `pipeline()` は、複数のモダリティをサポートしています。
    ここでは、テキスト処理と画像処理を組み合わせて、画像からインボイス番号を推論させています。

    <img src="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png" width="30%">
    """)
    return


@app.cell
def _(pipeline):
    # model: "impira/layoutlm-document-qa" (128M params)
    # ref: https://huggingface.co/impira/layoutlm-document-qa

    pipe_dqa = pipeline(model="impira/layoutlm-document-qa")
    out_dqa1 = pipe_dqa(
        image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
        question="What is the invoice number?",
    )
    out_dqa1[0]["score"] = round(out_dqa1[0]["score"], 3)
    out_dqa1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using pipeline on large models with 🤗 accelarate

    `device_map="auto"` を指定して、モデルを利用可能なデバイス上で適切に分配してロードします。
    これにより、単一のデバイスではメモリに乗り切らない大規模なモデルを利用することができます。
    """)
    return


@app.cell
def _(pipeline, torch):
    # model: "facebook/opt-1.3b" (1.3B params, heavy)
    # ref: https://huggingface.co/facebook/opt-1.3b

    pipe_acc1 = pipeline(model="facebook/opt-1.3b", dtype=torch.bfloat16, device_map="auto")
    pipe_acc1("This is a cool example!", do_sample=True, top_p=0.95)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    さらに、`bitsandbytes` ライブラリをインストールの上、`load_in_8bit=True` を指定すれば、モデルを 8 bit で量子化して読み込むことができます。
    ただし、`bitsandbytes` は現状 linux と windows しかサポートしていません。
    """)
    return


@app.cell
def _():
    # pipe_8bit1 = pipeline(model="facebook/opt-1.3b", device_map="auto", model_kwargs={"load_in_8bit": True})
    # pipe_8bit1("This is a cool example!", do_sample=True, top_p=0.95)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
