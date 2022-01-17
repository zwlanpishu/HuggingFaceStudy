from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np


# 加载数据集
raw_datasets = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding=True,
        truncation=True,
    )


# 数据集预处理为Index
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)


# 得到最终数据集格式
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

# 定义网络模型和优化器
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=2
)

# 定义metric
metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# 使用Trainer进行训练
training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
