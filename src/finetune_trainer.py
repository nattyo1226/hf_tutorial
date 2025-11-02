from datasets import load_dataset
import evaluate
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

def main():
    # load dataset
    dataset = load_dataset("yelp_review_full")
    
    # tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    
    # load pretrained model
    model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
    
    # load metric
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    # set training arguments
    training_args = TrainingArguments(
        output_dir="test_trainer",
        eval_strategy="epoch",
    )
    
    # build trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    # train
    trainer.train()

if __name__ == "__main__":
    main()
