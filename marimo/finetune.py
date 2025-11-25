import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fine-tune a pretrained model

    本チュートリアルでは、事前学習済みモデルに対してファインチューニングを行う方法を学びます。
    事前学習済みモデルに対して、特定のタスクに合わせたデータセットで追加の学習を行得ことで、ドメイン特化のモデルを得ることが期待できます。

    本チュートリアルは、[Hugging Face Tranformers チュートリアル](https://huggingface.co/docs/transformers/v4.57.1/ja/training) を元に、一部加筆・修正して作成しています。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dependencies

    このチュートリアルコードをすべて実行するためには、明示的に `import` するライブラリの他に必要なものは特にありません。
    """)
    return


@app.cell
def _():
    # run this cell if you are working in google colab

    # %pip install evaluate
    return


@app.cell
def _():
    from datasets import load_dataset
    import evaluate
    import numpy as np
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_scheduler,
        Trainer,
        TrainingArguments,
    )
    import torch
    from torch.nn import functional as F
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    return (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        F,
        Trainer,
        TrainingArguments,
        evaluate,
        load_dataset,
        np,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prepare a dataset

    今回は、[google-bert/bert-base-cased](https://huggingface.co/google-bert/bert-base-cased) (BERT) という Masked Language Model (MLM) をファインチューンしてみます。
    MLM とは、文章中の隠された (masked) 部分に当てはまる単語を推測するタスクを行う言語モデルです。
    BERT は [BookCorpus](https://yknzhu.wixsite.com/mbweb) と呼ばれる、11308 冊の未出版書籍と、英語の Wikipedia によって事前学習が行われています。

    今回は、BERT に対して、[yelp_review_full](https://huggingface.co/datasets/Yelp/yelp_review_full) というデータセットでファインチューニングを行います。
    このデータセットは、飲食店や店舗のレビューサイトである Yelp 上のレビュー文章から構成されています。
    早速、データセットをロードしましょう。
    """)
    return


@app.cell
def _(load_dataset):
    dataset = load_dataset("yelp_review_full")
    return (dataset,)


@app.cell
def _(dataset):
    dataset["train"].features
    return


@app.cell
def _(dataset):
    dataset["train"][100]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    データセットがロードできたら、トーカナイザをロードして事前処理を行いましょう。
    """)
    return


@app.cell
def _(AutoTokenizer, dataset):
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return (tokenized_datasets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    実行時間短縮のために、データセットから適当な部分セットを作成できます。
    """)
    return


@app.cell
def _(tokenized_datasets):
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    return small_eval_dataset, small_train_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train

    追加データセットの準備ができたので、ここから本格的にファインチューニングを行っていきます。
    ファインチューニングを行う手法にはいくつかあり、代表的なものは以下の 3 通りです。

    - `🤗 Transformers` が提供する `Trainer` クラスを利用するもの
    - `Kelas` API を利用して `TensorFlow` で訓練するもの
    - ネイティブの `PyTorch` で訓練するもの

    これらの利点・欠点は下表のとおりです (`GPT-5` および `gemini-2.5-pro` による回答のまとめ) 。

    | method | advantage 👍 | disadvantage 👎 |
    | :---- | :---- | :---- |
    | `Trainer` | 高水準 API を用いて短いコードで実装できる | 柔軟性に欠ける |
    | `Kelas` + `TensorFlow` | `TensorFlow` に慣れている人には親しみやすい、`Trainer` よりは柔軟に実装できる | `🤗 Transformers` はそもそも `PyTorch` 中心、コミュニティの規模が `PyTorch` に比べて小さい |
    | Native `PyTorch` | 低水準 API を用いて柔軟にカスタマイズできる | コード量が多く、実装が比較的複雑 |

    本チュートリアルでは、`Trainer` と Native `PyTorch` の 2 通りの手法を紹介します。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Train with PyTorch Trainer

    `Trainer` クラスを用いたファインチューニングの手順を紹介します。
    `🤗 Transformers` が提供する高水準 API を用いて、数行のプログラムで簡潔に記述することができます。

    まずモデルをロードし、予想される (マスクされる) ラベルの数を指定します。
    """)
    return


@app.cell
def _(AutoModelForSequenceClassification):
    model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
    model
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Training Hyperparameters

    学習時のオプションとハイパーパラメータから構成される `TrainingArguments` クラスを作成します。
    学習時のオプションとは、例えば学習後のパラメータファイルの保存先や、損失関数の値のログのタイミングなどを含みます。
    また、ハイパーパラメータとは、学習率のスケジューラやエポック数などを含みます。
    何も指定しなければ、デフォルトの値が利用されます。

    ```python
    training_args = TrainingArguments(output_dir="test_trainer")
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Evaluate

    `Trainer` は、デフォルトではモデルのパフォーマンスを評価しません。
    モデルのパフォーマンス評価を行うには、メトリクスを計算して報告する関数を `Trainer` に渡す必要があります。
    `🤗 Evaluate` ライブラリでは、`evaluate.load` 関数を使用して読み込むことができる、`accuracy` 関数が用意されています。
    """)
    return


@app.cell
def _(evaluate):
    metric = evaluate.load("accuracy")
    return (metric,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `metric.compute()` を呼ぶことで、予測精度を計算することができます。
    なお、すべての `🤗 Transformers` モデルの出力は logit だそうです。
    """)
    return


@app.cell
def _(metric, np):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    return (compute_metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    評価メトリクスをファインチューニング中に計算したい場合、学習引数 `eval_strategy` を利用できます。
    今回は、各エポック終了時に計算するように設定します。
    """)
    return


@app.cell
def _(TrainingArguments):
    training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")
    return (training_args,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Trainer

    モデル、学習引数、トレーニング/テストデータセット、評価メトリクスを指定して、`Trainer` オブジェクトを作成します。
    """)
    return


@app.cell
def _(
    Trainer,
    compute_metrics,
    model,
    small_eval_dataset,
    small_train_dataset,
    training_args,
):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )
    return (trainer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Trainer.train()` を実行して、ファインチューニングが行われます。
    """)
    return


@app.cell
def _(trainer):
    trainer.train()
    return


@app.cell
def _(AutoModelForSequenceClassification):
    model_trained = AutoModelForSequenceClassification.from_pretrained("./test_trainer/checkpoint-375")
    return (model_trained,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    テストしてみましょう。
    """)
    return


@app.cell
def _(F, model, model_trained, small_train_dataset, torch):
    id = 0  # 0:1000
    dataset_tmp = small_train_dataset  # train/eval

    input_tmp = {
        "input_ids": torch.tensor(dataset_tmp["input_ids"][id]).unsqueeze(0),
        "token_type_ids": torch.tensor(dataset_tmp["token_type_ids"][id]).unsqueeze(0),
        "attention_mask": torch.tensor(dataset_tmp["attention_mask"][id]).unsqueeze(0),
    }
    label_tmp = dataset_tmp["label"][id]

    print(label_tmp)
    print(F.softmax(model(**input_tmp).logits, dim=1))
    print(F.softmax(model_trained(**input_tmp).logits, dim=1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Train in native Pytorch

    次に、ネイティブ `PyTorch` でファイチューニングを行う手順を紹介します。
    複雑ではあるものの、カスタマイズ性の高い学習ループを構成できます。
    なお、以下の部分は上で定義した変数と衝突するため、`marimo` でチュートリアルを実行している方は、一度セッションを切って、`Trainer` API による実装部分をコメントアウトしてから、以下のプログラムを実行してください。

    まずデータセットのロードを行うのですが、

    1. モデルはトークン化前のオリジナルテキストを受け取らないので、`text` 列を削除する。
    2. モデルは引数の名前を `labels` と期待しているので、`label` 列を `labels` に名前を変更する。

    という前処理を行います。
    """)
    return


@app.cell
def _():
    # run this cell if you are working on ipynb (google colab)

    # del model
    # del trainer
    # torch.cuda.empty_cache()
    return


@app.cell
def _():
    # tokenized_datasets_in_need = tokenized_datasets.remove_columns(["text"])
    # tokenized_datasets_pt = tokenized_datasets_in_need.rename_column("label", "labels")
    # tokenized_datasets_pt.set_format("torch")

    # small_train_dataset_pt = tokenized_datasets_pt["train"].shuffle(seed=42).select(range(1000))
    # small_eval_dataset_pt = tokenized_datasets_pt["test"].shuffle(seed=42).select(range(1000))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### DataLoader

    トレーニングデータセットとテストデータセット用の `DataLoader` を作成して、データのバッチをイテレータとして取り出せるようにします。
    """)
    return


@app.cell
def _():
    # train_dataloader = DataLoader(small_train_dataset_pt, shuffle=True, batch_size=8)
    # eval_dataloader = DataLoader(small_eval_dataset_pt, batch_size=8)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    併せて、モデルのロードも行ってしまいます。
    やり方は `Trainer` の場合と同じです。
    """)
    return


@app.cell
def _():
    # model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Opeimizer and learning rate scheduler

    オプティマイザと学習率スケジューラを作成します。
    ここでは、`AdamW` を用いてモデルの最適化を行うことにします。
    """)
    return


@app.cell
def _():
    # optimizer = AdamW(model.parameters(), lr=5e-5)
    return


@app.cell
def _():
    # num_epochs = 3
    # num_training_steps = num_epochs * len(train_dataloader)
    # lr_scheduler = get_scheduler(
    #     name="linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=num_training_steps,
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    また、ファインチューニングを行うデバイスを指定しておきましょう。
    """)
    return


@app.cell
def _():
    # for NVIDIA GPU (CUDA)
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # for Apple GPU (MPS)
    # device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # model.to(device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Train

    学習の進捗を追跡するために、`tqdm` ライブラリを使用して進行状況バーを表示させます。
    """)
    return


@app.cell
def _():
    # progress_bar = tqdm(range(num_training_steps))

    # model.train()
    # for epoch in range(num_epochs):
    #     for batch_t in train_dataloader:
    #         batch_train = {k: v.to(device) for k, v in batch_t.items()}
    #         outputs_train = model(**batch_train)
    #         loss = outputs_train.loss
    #         loss.backward()

    #         optimizer.step()
    #         lr_scheduler.step()
    #         optimizer.zero_grad()
    #         progress_bar.update(1)

    # model.save_pretrained("test_torch")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Evaluate

    `Trainer` の際と同様に、評価メトリックを導入します。
    ここでは、各エポックの最後にメトリックを計算する代わりに、`add_batch` を使用してすべてのバッチを蓄積しておき、最後にメトリックを計算することにします。
    """)
    return


@app.cell
def _():
    # metric = evaluate.load("accuracy")
    # model.eval()
    # for batch_e in eval_dataloader:
    #     batch_eval = {k: v.to(device) for k, v in batch_e.items()}
    #     with torch.no_grad():
    #         outputs_eval = model(**batch_eval)

    #     logits = outputs_eval.logits
    #     predictions = torch.argmax(logits, dim=-1)
    #     metric.add_batch(predictions=predictions, references=batch["labels"])

    # metric.compute()
    return


@app.cell
def _():
    # model_trained = AutoModelForSequenceClassification.from_pretrained("./test_torch")
    return


@app.cell
def _():
    # id = 0  # 0:1000
    # dataset_tmp = small_train_dataset_pt  # train/eval

    # input_tmp = {
    #     "input_ids": dataset_tmp["input_ids"][id].unsqueeze(0),
    #     "token_type_ids": dataset_tmp["token_type_ids"][id].unsqueeze(0),
    #     "attention_mask": dataset_tmp["attention_mask"][id].unsqueeze(0),
    # }
    # label_tmp = dataset_tmp["labels"][id]

    # print(label_tmp)
    # print(F.softmax(model(**input_tmp).logits, dim=1))
    # print(F.softmax(model_trained(**input_tmp).logits, dim=1))
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
