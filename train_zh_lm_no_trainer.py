import os
import argparse
import math
import random
import datasets
from tqdm import tqdm
from itertools import chain
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    set_seed,
)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from logger import LanguageModelingLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a transformer model on a Masked Language Modeling task"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=None,
        help="The default args from torch.distributed.lanch",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="/home/server/disk1/DATA/wiki_zh_2019/train.txt",
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default="/home/server/disk1/DATA/wiki_zh_2019/valid.txt",
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/home/server/disk1/checkpoints/HuggingFaceStudy/baseline/epoch_0",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=6,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=6,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--save_interval", type=int, default=2, help="The interval of epochs to save the model.",
    )
    parser.add_argument(
        "--restore", type=bool, default=True, help="Wheather to restore the training model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/server/disk1/checkpoints/HuggingFaceStudy/baseline",
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/server/disk1/checkpoints/HuggingFaceStudy/log/baseline",
        help="Where to store the log information.",
    )
    args = parser.parse_args()
    return args


def sanity_check(args):
    assert args.local_rank is not None, "Distributed launch script is needed."
    assert args.dataset_name is not None or args.train_file is not None
    return


def prepare_tensorboard(args):
    if args.local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        os.chmod(args.output_dir, 0o775)
        os.makedirs(args.log_dir, exist_ok=True)
        os.chmod(args.log_dir, 0o775)
        logger = LanguageModelingLogger(args.log_dir)
    else:
        logger = None
    return logger


def main():
    args = parse_args()
    # TODO: 检查参数, 加入断点恢复, 增加日志
    sanity_check(args)
    set_seed(1234)
    logger = prepare_tensorboard(args)
    dist.init_process_group(backend="nccl", init_method="env://")

    # 0. Load the dataset from the hub or custom file.
    if args.dataset_name is not None:
        # from the HuggingFace hub
        raw_datasets = datasets.load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = datasets.load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = datasets.load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        # from custom file
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"

        raw_datasets = datasets.load_dataset(extension, data_files=data_files)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = datasets.load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = datasets.load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
            )

    # 1. Load the tokenizer.
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError("A tokenizer is needed.")

    # 2. Load the config and the model.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError("A config is needed.")

    if args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path, config=config,)
    else:
        # Train a model from scratch based on a configuration file.
        model = AutoModelForMaskedLM.from_config(config)
    model.resize_token_embeddings(len(tokenizer))
    print(f"GPU：{args.local_rank} is ready for the dataset, tokenizer and model")

    # 3. Preprocess for the dataset.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    tokenized_datasets = raw_datasets.map(
        tokenize_function, batched=True, remove_columns=[text_column_name], num_proc=8,
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= args.max_seq_length:
            total_length = (total_length // args.max_seq_length) * args.max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + args.max_seq_length] for i in range(0, total_length, args.max_seq_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=8)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=args.mlm_probability
    )
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size,
    )
    for index in random.sample(range(len(train_dataset)), 1):
        print(
            f"GPU：{args.local_rank}, Sample {index} of the training set: {train_dataset[index]}."
        )

    # 4. Create the optimizer and learning rate scheduler.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(elem in n for elem in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(elem in n for elem in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.98)

    # Assign the max train steps or number of train epochs.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 5. Train loop begin
    if args.local_rank == 0:
        print("***** Running training *****")
        print(f"Num examples = {len(train_dataset)}")
        print(f"Num Epochs = {args.num_train_epochs}")
        print(f"Batch size per device = {args.per_device_train_batch_size}")
        print(f"Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"Total optimization steps = {args.max_train_steps}")

    if args.restore is True:
        print(f"GPU {args.local_rank} restore the training model from {args.model_name_or_path}")
        assert os.path.isdir(args.model_name_or_path) is True, f"GPU {args.local_rank} load error"
        restore_file = os.path.join(args.model_name_or_path, "restore.pt")
        assert os.path.exists(restore_file) is True, f"GPU {args.local_rank} load error"
        restore_dict = torch.load(restore_file, map_location="cpu")
        completed_steps = restore_dict["completed_steps"]
        # The epoch should begin for next
        completed_epoch = restore_dict["completed_epochs"] + 1
        optimizer = optimizer.load_state_dict(restore_dict["optimizer"])
        lr_scheduler = lr_scheduler.load_state_dict(restore_dict["lr_scheduler"])
    else:
        completed_steps = 0
        completed_epoch = 0

    device = (
        torch.device(f"cuda:{args.local_rank}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[args.local_rank])

    for epoch in range(completed_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.local_rank == 0:
                    print(f"Epoch: {epoch}, Completed_steps: {completed_steps}, Loss: {loss}")
                    logger.log_train(loss, lr_scheduler.get_last_lr()[0], completed_steps)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                print("Training is finished.")
                break

        if args.local_rank == 0:
            model.eval()
            losses = []
            for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                losses.append(loss.repeat(args.per_device_eval_batch_size))

            losses = torch.cat(losses)
            losses = losses[: len(eval_dataset)]
            try:
                perplexity = math.exp(torch.mean(losses))
            except OverflowError:
                perplexity = float("inf")
            logger.log_eval(perplexity, epoch)
            print(f"Epoch: {epoch}, perplexity: {perplexity}")

            # save the model
            if epoch % args.save_interval == 0 or epoch == len(args.num_train_epochs - 1):
                model.module.save_pretrained(os.path.join(args.output_dir, f"epoch_{epoch}"))
                tokenizer.save_pretrained(os.path.join(args.output_dir, f"epoch_{epoch}"))
                torch.save(
                    {
                        "completed_steps": completed_steps,
                        "completed_epochs": epoch,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    os.path.join(args.output_dir, f"epoch_{epoch}", "restore.pt"),
                )


if __name__ == "__main__":
    main()
