import datasets
import glob
import os
from transformers import AutoTokenizer
from functools import partial
import torch
from torch.utils.data import DataLoader
def load_dataset(data_path,sft_split="train"):
    all_jsonl_files = glob.glob(os.path.join(data_path, "*.jsonl"))
    return datasets.load_dataset("json", data_files=all_jsonl_files)[sft_split]

_global_tokenizer = None
def tokenize_fn(examples,tokenizer_dir,eos_id=0,pad_id=0):
    global _global_tokenizer
    if _global_tokenizer is None:
        _global_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,trust_remote_code=True)
    tokenizer = _global_tokenizer
    all_input_ids = []
    all_labels = []
    input_texts = examples["input"]
    output_texts = examples["output"]
    for input_text,output_text in zip(input_texts,output_texts):
        input_ids = tokenizer.encode(input_text)
        output_ids = tokenizer.encode(output_text)
        whole_inputs = input_ids + output_ids
        labels = whole_inputs.copy()
        labels[:len(input_ids)] = [-100] * len(input_ids)
        labels=labels[1:]+[eos_id]
        all_input_ids.append(whole_inputs)
        all_labels.append(labels)
    return {"input_ids": all_input_ids, "labels": all_labels}

def data_collator(examples,pad_id=0):
    input_ids, labels = tuple([instance[key] for instance in examples] for key in ("input_ids", "labels"))
    input_ids = [torch.tensor(x) for x in input_ids]
    attention_mask = [torch.tensor([1] * len(x)) for x in input_ids]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id,padding_side="left")
    labels = [torch.tensor(x) for x in labels]
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100,padding_side="left")
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0,padding_side="left")
    return input_ids, labels,  attention_mask

if __name__ == "__main__":
    tokenizer_dir = "/home/zebraclips/yynil/models/rwkv7-1.5B-g1a/"
    dataset = load_dataset("demo_data/translation",sft_split="train")
    dataset = dataset.map(partial(tokenize_fn,tokenizer_dir=tokenizer_dir,eos_id=0,pad_id=0),batched=True,remove_columns=dataset.column_names)
    print(dataset)
    print(dataset[0])
    dataloader = DataLoader(dataset,batch_size=8,collate_fn=partial(data_collator,pad_id=0),shuffle=True)
    for batch in dataloader:
        print(batch)
        input_ids, labels, attention_mask = batch
        print(input_ids.shape)
        print(labels.shape)
        print(attention_mask.shape)
        break