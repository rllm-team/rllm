import os
from dataclasses import dataclass
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm


def get_hf_dataset(csv_file: str, target_column: str):
    from datasets import Dataset, ClassLabel, Value, Features, load_from_disk

    # Read the input CSV file
    df = pd.read_csv(csv_file)

    # Get the directory path of the CSV file to ensure the output folder is created in the same directory
    base_dir = os.path.dirname(os.path.abspath(csv_file))
    dataset_name = os.path.basename(csv_file).split(".")[
        0
    ]  # Get the name of the dataset from the CSV file name
    output_path = os.path.join(
        base_dir, "output_dataset", dataset_name
    )  # Default folder name is 'output_dataset'

    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        # If the directory already exists, load the dataset from disk
        ds = load_from_disk(output_path)
        return ds

    # Combine all columns except the target_column into a single string
    other_columns = [col for col in df.columns if col != target_column]

    def create_note(row):
        note_parts = [f"The {col} is {row[col]}." for col in other_columns]
        return " ".join(note_parts)

    df["note"] = df.apply(create_note, axis=1)

    # Create a new column for the target_column, named text_label
    df["text_label"] = df[target_column]

    # Map the target_column to 0, 1, 2 and store the mapping
    unique_labels = df[target_column].unique()
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    df["label"] = df[target_column].map(label_mapping)

    # Create a new column idx, starting from 0 and increasing sequentially
    df["idx"] = df.index

    # Define features
    label_class = ClassLabel(
        names=[str(label) for label in unique_labels]
    )  # Convert category names to string

    # Use Features to explicitly specify the data types of each column
    features = Features(
        {
            "note": Value("string"),
            "text_label": Value("string"),
            "label": label_class,
            "idx": Value("int32"),
        }
    )

    # Create a Hugging Face dataset
    ds = Dataset.from_pandas(
        df[["note", "text_label", "label", "idx"]], features=features
    )

    # Save the dataset to the specified directory
    ds.save_to_disk(output_path)

    # Return the path where the dataset is saved
    # return output_path, ds
    return ds


# —— 1. Configuration Class —— #
@dataclass
class FinetuneConfig:
    csv_dataset: str  # csv file path
    target_column: str  # target column name
    task_info_path: str  # task info path
    test_size: float = 0.8
    tokenizer_name: str = "google-t5/t5-small"
    model_name: str = "google-t5/t5-small"
    peft_task_type: str = "SEQ_2_SEQ_LM"
    lr: float = 8e-3
    batch_size: int = 8
    num_epochs: int = 3
    warmup_steps: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# —— 2. Internal Tools: Dataset + Collator —— #
class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, task_info, add_special_tokens=True):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.task_info = task_info
        self.add_special_tokens = add_special_tokens

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        example = self.dataset[key]
        input_str = example["note"] + "\n\n\n" + self.task_info + "\nAnswer:"
        target_str = example["text_label"]
        answer_choices = self.dataset.features["label"].names

        input_ids = self.tokenizer(
            input_str,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=self.add_special_tokens,
        ).input_ids.squeeze(0)
        target_ids = self.tokenizer(
            target_str,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=self.add_special_tokens,
        ).input_ids.squeeze(0)
        answer_choices_ids = [
            self.tokenizer(
                answer_choice,
                return_tensors="pt",
                truncation=True,
                add_special_tokens=self.add_special_tokens,
            ).input_ids.squeeze(0)
            for answer_choice in answer_choices
        ]
        label = torch.LongTensor([example["label"]])
        idx = torch.LongTensor([example["idx"]])
        return input_ids, target_ids, answer_choices_ids, label, idx


def create_collate_fn(pad_token_id):
    def collate_fn(batch):
        input_ids, target_ids, answer_choices_ids, labels, idx = zip(*batch)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
        attention_mask = (input_ids != pad_token_id).long()
        target_ids = torch.nn.utils.rnn.pad_sequence(
            target_ids, batch_first=True, padding_value=pad_token_id
        )
        target_ids[target_ids == pad_token_id] = (
            -100
        )  # Mask the padding token in the target sequence
        output_batch = {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask,
        }
        return output_batch

    return collate_fn


# —— 3. High-Level Wrapper —— #
class Seq2SeqFinetuner:
    def __init__(self, cfg: FinetuneConfig):
        self.cfg = cfg
        # tokenizer + split
        self.raw_dataset = get_hf_dataset(cfg.csv_dataset, cfg.target_column)
        # print(type(self.raw_dataset))
        ds = self.raw_dataset.train_test_split(test_size=cfg.test_size)
        ds["validation"] = ds["test"]
        del ds["test"]
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        assert cfg.task_info_path.endswith(
            ".txt"
        ), "task_info_path should be a .txt file"
        with open(cfg.task_info_path, "r", encoding="utf-8") as f:
            self.task_info = f.read().strip()
        self.train_ds = FinetuneDataset(ds["train"], self.tokenizer, self.task_info)
        self.val_ds = FinetuneDataset(ds["validation"], self.tokenizer, self.task_info)

        # model + PEFT
        from peft import IA3Config, get_peft_model
        base = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
        peft_config = IA3Config(task_type=cfg.peft_task_type)
        self.model = get_peft_model(base, peft_config).to(cfg.device)
        self.model.print_trainable_parameters()

    def train(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id),
            drop_last=True,
        )
        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id),
        )
        total_steps = (len(self.train_ds) // self.cfg.batch_size) * self.cfg.num_epochs
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=total_steps,
        )

        for epoch in range(1, self.cfg.num_epochs + 1):
            # —— train —— #
            self.model.train()
            total = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch} train"):
                batch = {k: v.to(self.cfg.device) for k, v in batch.items()}
                out = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["target_ids"],
                )
                loss = out.loss
                total += loss.detach().float()
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            print(f"  Train loss: {total/len(train_loader):.4f}")

            # —— validation —— #
            self.model.eval()
            total = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch} val"):
                    batch = {k: v.to(self.cfg.device) for k, v in batch.items()}
                    out = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["target_ids"],
                    )
                    total += out.loss.detach().float()
            print(f"  Val   loss: {total/len(val_loader):.4f}")

    def predict(self, inputs: List[str]) -> List[str]:
        """Simple inference interface that returns a list of strings."""
        prompts = [inp + "\n\n\n" + self.task_info + "\nAnswer:" for inp in inputs]
        print(prompts)
        with torch.no_grad():
            self.model.eval()
            # Generate
            tid = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            ).to(self.cfg.device)
            out = self.model.generate(**tid, max_new_tokens=10)
            return self.tokenizer.batch_decode(
                out.detach().cpu().numpy(), skip_special_tokens=True
            )


if __name__ == "__main__":
    # Example usage
    cfg = FinetuneConfig(
        csv_dataset="Your CSV file path here",
        target_column="Your target column name here",
        task_info_path="Your task info file path(txt format) here",
    )
    finetuner = Seq2SeqFinetuner(cfg)
    finetuner.train()
