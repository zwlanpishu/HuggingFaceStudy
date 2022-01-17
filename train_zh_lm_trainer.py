import datasets
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling


# 0. 定义使用的中文数据集
wiki_zh = datasets.load_dataset(
    "text",
    data_files={
        "train": "/home/server/disk1/DATA/wiki_zh_2019/wiki_zh_clean/train.txt",
        "validation": "/home/server/disk1/DATA/wiki_zh_2019/wiki_zh_clean/valid.txt",
    },
)


# 1. 定义使用的分词器，这里与Google Bert保持一致
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", do_lower_case=True)


def tokenize_function(examples):
    return tokenizer(examples["text"])


tokenized_datasets = wiki_zh.map(
    tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
)


# 2. 将数据集中的句子切分为指定的长度
# block_size = tokenizer.model_max_length
block_size = 128


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

# 3. 定义模型结构
config = AutoConfig.from_pretrained("bert-base-chinese")
model = AutoModelForMaskedLM.from_config(config)

# 4. 定义训练使用的参数
training_args = TrainingArguments(
    "/home/server/disk1/checkpoints/bert-base-chinese-repro",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2.4e-4,
    weight_decay=0.01,
    num_train_epochs=500,
    save_strategy="epoch",
    save_steps=1,
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=0.15
)

# 5. 开始模型训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,

)
trainer.train()
