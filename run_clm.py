token="breadlicker45/muse-tokenizer"
model="decapoda-research/llama-7b-hf"
dataset_name="breadlicker45/musenet-chunk"
collum='bing'
max_length=2048
learning=3e-6

import torch
from torch.utils.data import Dataset, random_split
from torch.nn import DataParallel
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, AdamW, TrainingArguments
from collie import Trainer, LlamaForCausalLM, CollieConfig
from datasets import load_dataset
import numpy as np
from accelerate import Accelerator

accelerator = Accelerator()

#loading model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(token, bos_token='<|startoftext|>',
                                          eos_token='<|endoftext|>', pad_token='<|pad|>')
config = CollieConfig.from_pretrained("decapoda-research/llama-7b-hf")


config.ds_config = {
    "fp16": {
        "enabled": True
    },
    # "monitor_config": {
    #     "enabled": True,
    #     "wandb": {
    #         "enabled": True,
    #         "team": "00index",
    #         "project": "collie",
    #         "group": "test_evaluator"
    #     }
    # },
    "zero_optimization": {
        "stage": 3,
    }
}


model = LlamaForCausalLM.from_pretrained(model, config=config)
#model.resize_token_embeddings(len(tokenizer))
optimizer = AdamW(model.parameters(), lr=learning)
#model = DataParallel(model, device_ids=[0, 1])
model, optimizer = accelerator.prepare(
     model, optimizer
 )

import transformers
import collie
from datasets import load_dataset
dataset_name=dataset_name
data = load_dataset(dataset_name)
data = data.map(lambda samples: tokenizer(samples[collum], padding='max_length', truncation=True, max_length=max_length), batched=True)


trainer = collie.Trainer(
    model=model, 
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        auto_find_batch_size=True, 
        warmup_ratio=0.05,
        num_train_epochs=1, 
        learning_rate=learning, 
        lr_scheduler_type="cosine",
        fp16=True,
        save_steps=15500,
        logging_steps=1, 
        report_to="wandb",
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
CUDA_LAUNCH_BLOCKING=1 
trainer.train()
model.save_pretrained("save/")
