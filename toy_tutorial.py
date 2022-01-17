from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import (
    DistilBertConfig,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)
from torch import nn
import torch

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pt_batch = tokenizer(
    [
        "We are very happy to show you the 🤗 Transformers library.",
        "We hope you don't hate it.",
    ],
    truncation=True,
    max_length=512,
)
print(pt_batch[0])

for key, value in pt_batch.items():
    print(f"{key}: {value.numpy().tolist()}")

pt_outputs = pt_model(
    **pt_batch,
    labels=torch.tensor([1, 0]),
    output_hidden_states=True,
    output_attentions=True,
)
pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
print(pt_outputs)

test = {"a": 1, "b": 3}
pt_save_directory = "./pt_save_pretrained"
tokenizer.save_pretrained(pt_save_directory)
pt_model.save_pretrained(pt_save_directory)

all_hidden_states = pt_outputs.hidden_states
all_attentions = pt_outputs.attentions

config = DistilBertConfig(n_heads=8, dim=512, hidden_dim=4 * 512)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification(config)
