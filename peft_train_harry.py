import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
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

def check_str(x):
    #x = re.findall('[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\u4e00-\u9fa5\w]',x)
    #x = ''.join(x)
    ##x = x.replace('\u3000', '')
    ##x = x.replace('\xa0', '')
  

    return {'sentence': x}

device = "cuda"
## bigscience/T0_3B, bigscience/mt0-large, bigscience/mt0-small, THUDM/chatglm-6b
model_name_or_path = "bigscience/bloomz-1b7" 

lr = 5e-5
num_epochs = 10
batch_size = 4

# creating model
continue_train = False
peft_model_id = 'checkpoint/bigscience/bloomz-1b7_LORA_20230814'
model, peft_config = create_model(continue_train, model_name_or_path, peft_model_id)#, load_8bit_flag=True)

## read data
with open('data/harrypotter_train_text.txt') as f:
  data = f.readlines()

#data = list(filter(lambda x: len(x.strip()) >= 30, data))
data = list(map(check_str, data))

## tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
dataset = SentenceDataset(data, tokenizer)

train_size = int(len(dataset) * 0.8)
valid_size = len(dataset) - train_size

train_set, valid_set = random_split(dataset, [train_size, valid_size], 
                                    generator=torch.Generator().manual_seed(42))


train_dataloader = DataLoader(
    train_set, shuffle=True, #collate_fn=default_data_collator, 
    batch_size=batch_size, pin_memory=True,
    num_workers=4,
    collate_fn=default_data_collator,
    prefetch_factor=2
)

valid_dataloader = DataLoader(
    valid_set, shuffle=False, #collate_fn=default_data_collator, 
    batch_size=batch_size, pin_memory=True,
    num_workers=4,
    collate_fn=default_data_collator,
    prefetch_factor=2
)

print('Total sentences:', len(data), flush=True)
print('train steps', len(train_set) // batch_size, 'valid steps', len(valid_set), flush=True)

# optimizer and lr scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

# training and evaluation
model = model.to(device)

chk_folder = 'checkpoint'
os.makedirs(chk_folder, exist_ok=True)

for epoch in range(num_epochs):
    if epoch > 0:
        total_loss = 0
        model.train()
        st = time.time()
        #for step, batch in enumerate(tqdm(train_dataloader)):
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        et = time.time()
        train_time = '%.2f'%(et - st)

    if (epoch+1) % 2 == 0:
        # saving checkpoints
        time_now = datetime.now().strftime('%Y%m%d%H%M')
        peft_model_id = f"{chk_folder}/{model_name_or_path}_{peft_config.peft_type}_{time_now}"
        model.save_pretrained(peft_model_id)

    model.eval()
    eval_loss = 0
    eval_preds = []
    #for step, batch in enumerate(tqdm(valid_dataloader)):
    print('Start to do the evaluate...')
    for step, batch in enumerate(valid_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )
        if step == 0:
            ins  = np.squeeze(batch['input_ids'][0].cpu().numpy())
            outs = np.squeeze(batch['labels'][0].cpu().numpy())
            outs[outs == -100] = 0

            sample_input = tokenizer.decode(ins, skip_special_tokens=True)
            sample_output = tokenizer.decode(outs, skip_special_tokens=True)


    if epoch > 0: 
        eval_epoch_loss = eval_loss / len(valid_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"===== EPOCH:{epoch} ==============")
        print(f"train_ppl:{train_ppl:.2f} train_loss:{train_epoch_loss:.2f}")
        print(f"eval_ppl:{eval_ppl:.2f} eval_loss:{eval_epoch_loss:.2f}")
        print('Train time:', train_time)

    print('One Sample Sentence')
    print('input:', sample_input, len(sample_input))
    print('label:', sample_output)
    print('prediction:', eval_preds[0], flush=True)

print('Training Done')
