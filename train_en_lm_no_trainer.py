from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader


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
tokenized_datasets.set_format("torch")

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

# 定义网络模型和优化器
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=2
)
optimizer = AdamW(model.parameters(), lr=5e-5)

# 定义metric
metric = load_metric("accuracy")

# 定义Pytorch Dataloader
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


# 开始模型的训练
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


# 开始模型的验证
metric = load_metric("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
print(metric)
