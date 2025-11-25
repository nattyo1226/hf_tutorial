from datasets import load_dataset
import evaluate
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def main():
    # load dataset
    dataset = load_dataset("yelp_review_full")
    
    # tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)    
    tokenized_datasets_in_need = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets_pt = tokenized_datasets_in_need.rename_column("label", "labels")
    tokenized_datasets_pt.set_format("torch")

    small_train_dataset_pt = tokenized_datasets_pt["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset_pt = tokenized_datasets_pt["test"].shuffle(seed=42).select(range(1000))
    
    train_dataloader = DataLoader(small_train_dataset_pt, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(small_eval_dataset_pt, batch_size=8)
    
    # load pretrained model
    model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
    
    # build optimiser
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # build learning rate scheduler
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    # set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # train
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch_t in train_dataloader:
            batch_train = {k: v.to(device) for k, v in batch_t.items()}
            outputs_train = model(**batch_train)
            loss = outputs_train.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    
    model.save_pretrained("test_torch")
    
    # evaluate
    metric = evaluate.load("accuracy")
    model.eval()
    for batch_e in eval_dataloader:
        batch_eval = {k: v.to(device) for k, v in batch_e.items()}
        with torch.no_grad():
            outputs_eval = model(**batch_eval)

        logits = outputs_eval.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch_eval["labels"])
        
    print(metric.compute())

if __name__ == "__main__":
    main()
