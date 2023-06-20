token="breadlicker45/muse-tokenizer"
model="decapoda-research/llama-7b-hf"
dataset_name="breadlicker45/Calorie-dataset"
collum='Data'
max_length=2048
learning=3e-6

# main.py
import transformers
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from collie import Trainer, CollieConfig, LlamaForCausalLM
from collie.optim.lomo import Lomo
from transformers import LlamaTokenizer

config = CollieConfig.from_pretrained("breadlicker45/llama-test")
config.ds_config = {
    "fp16": {
        "enabled": True
    }
}
model = LlamaForCausalLM.from_pretrained("breadlicker45/llama-test", config=config)
optimizer = Lomo(model=model, lr=2e-5)
tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", add_eos_token=True)
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2


dataset_name=dataset_name
data = load_dataset(dataset_name) 
data = data.map(lambda samples: tokenizer(samples[collum], padding='max_length', truncation=True, max_length=max_length), batched=True)


trainer = Trainer(
    model=model,
    config=config,
    optimizer=optimizer,
    train_dataset=[({"input_ids": data["train"].input_ids}, {"labels": data["train"].input_ids})]
)
trainer.train()
model.save_pretrained("save/")
