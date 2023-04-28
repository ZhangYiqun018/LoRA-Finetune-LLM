# %%
from transformers import AutoTokenizer, PreTrainedTokenizerBase, DataCollatorForSeq2Seq, Seq2SeqTrainer, AutoModelForSeq2SeqLM
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_config, get_peft_model, get_peft_model_state_dict
import json
from typing import Dict

def get_config(
    config_path: str,
):
    config_file = open(config_path, 'r')
    config = json.load(config_file)
    return config

def pre_process(example, tokenizer: PreTrainedTokenizerBase):
    input  = example['input']
    output = example['output']
    
    example_token = tokenizer(
        text        = input,
        text_target = output,
        padding     = True,
        truncation  = 'only_first',
    )

    return example_token

def get_model(
    model_card: str,
    do_lora: bool,
    lora_config: Dict,
    use_8bit: bool,
):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_card, device_map="auto")
    if do_lora:
        print(" * * * do lora * * * ")
        lora_config.pop('do_lora')
        lora_config = LoraConfig(
            **lora_config,
        )
        model = get_peft_model(
            model,
            lora_config
        )
        model.print_trainable_parameters()
    return model

def get_tokenizer(
    model_card: str,
):
    tokenizer = AutoTokenizer.from_pretrained(model_card)
    tokenizer.sep_token = '<sep>'
    return tokenizer

def get_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizerBase,
    local: bool,
): 
    if local:
        dataset = load_from_disk(dataset_path=data_path)
    else:
        dataset = load_dataset(
            data_path,
        )
    
    dataset = dataset.map(
        lambda x: pre_process(x, tokenizer), batched=True, num_proc=8,
    )

    dataset = dataset.remove_columns(
        ['input', 'output']
    )

    if 'knowledge_type' in dataset.column_names:
        dataset = dataset.remove_columns(
            ['knowledge_type']
        )
    if 'task_type' in dataset.column_names:
        dataset = dataset.remove_columns(
            ['task_type']
        )

    return dataset


# %%
if __name__ == '__main__':
    model_card: str = '/datas/huggingface/Flan-T5/flan-t5-base'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_card)
    tokenizer.sep_token = '<sep>'

    dataset_card: str = 'Estwld/atomic-instruct-v1'
    dataset = get_dataset(data_path=dataset_card, tokenizer=tokenizer)
# %%
