from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from transformers import default_data_collator, get_linear_schedule_with_warmup
from datasets import load_dataset, Dataset

from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from peft import PeftModel, PeftConfig

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
from datetime import datetime
import numpy as np
import re, time

from data_load import SentenceDataset
from function import create_model

from peft import PeftModel, PeftConfig

def check_str(x):
  #x = re.findall('[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\u4e00-\u9fa5\w]',x)
  #x = ''.join(x)
  ##x = x.replace('\u3000', '')
  ##x = x.replace('\xa0', '')

  return {'sentence': x}

device = 'cuda'
peft_model_id = f"checkpoint/bigscience/bloomz-1b7_LORA_202308150125"
model_name_or_path = "bigscience/bloomz-1b7"

config = PeftConfig.from_pretrained(peft_model_id)
#model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)

## original pre-trained model
#model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
#model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

model.eval()
model = model.to(device)

## read data
with open('data/harrypotter_train_text.txt') as f:
  data = f.readlines()

data = list(map(check_str, data))

## tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
dataset = SentenceDataset(data, tokenizer)

train_size = int(len(dataset) * 0.8)
valid_size = len(dataset) - train_size

train_set, valid_set = random_split(dataset, [train_size, valid_size],
                                    generator=torch.Generator().manual_seed(42))

i = 3
inputs = valid_set[i]
label_ids  = valid_set[i]['labels'][0]
label_ids  = np.array(label_ids)
label_ids[label_ids == -100] = 0
#print(label_ids)


input_text = tokenizer.decode(inputs['input_ids'], skip_special_tokens=True)
label_texts = tokenizer.decode(label_ids)

orig_inputs = input_text
all_texts = ''
###
input_text = input("Input the sentence:")
#input_text = 'While the Prime Minister surreptitiously touched the wood of his desk'
inputs = tokenizer(input_text)

print('input text:', input_text)
input_len = len(input_text)
st = time.time()
for j in range(1):
    with torch.no_grad():
        #outputs = model.generate(input_ids=inputs['input_ids'].to(device), max_new_tokens=1024, temperature=0.8,
        #                       num_beams=3, repetition_penalty=1.3, do_sample=True)

        outputs = model.generate(input_ids=torch.tensor([inputs['input_ids']], dtype=torch.int).to(device), max_new_tokens=100, do_sample=True)
        output_text = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)

    
    all_texts = output_text[0][:-4]
    #print('output:', output_text[0])
    #print('=========================')
    input_text = all_texts
    inputs = tokenizer(input_text+ '->', return_tensors="pt", max_length=1024, padding="max_length")
et = time.time()

print('======================')
print(output_text[0][input_len:])
#output = output_text[0].replace('->', '')
#print(output)
