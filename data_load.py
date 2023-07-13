import torch
import numpy as np

class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, tokenizer, block_size=128):
        token_data = [tokenizer(i['sentence']) for i in sentences]

        all_token = {i: [] for i in token_data[0].keys()}
        for item in token_data:
            for key in all_token:
                all_token[key].extend(item[key])

        self.all_token = all_token
        self.sentences = sentences
        
        self.nums = len(self.all_token[list(self.all_token.keys())[0]]) // block_size
        self.tokenizer = tokenizer

        self.block_size = block_size

    def __getitem__(self, idx):
        model_inputs = {
            k: t[idx*self.block_size: (idx+1)*self.block_size]
            for k, t in self.all_token.items()
        }

        model_inputs['labels'] = model_inputs['input_ids'].copy()

        return model_inputs

    def __len__(self):
        return self.nums

if __name__ == '__main__':
    from transformers import AutoTokenizer
    import torch
    import re
    from torch.utils.data import DataLoader, random_split
    import json

    def check_str(x):
        x = re.findall('[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\u4e00-\u9fa5\w]',x)
        x = ''.join(x)
        #x = x.replace('\u3000', '')
        #x = x.replace('\xa0', '')
        
        return {'sentence': x + '\n'}

    model_name_or_path = "bigscience/mt0-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    ## read data
    with open('data/天龍八部.txt') as f:
      data = f.readlines()[40:]
    
    data = list(filter(lambda x: len(x.strip()) >= 30, data))
    data = list(map(check_str, data))
    #print('length:', len(data))
    #print(data[:2])

    dataset = SentenceDataset(data[:10], tokenizer)

    train_size = int(len(dataset) * 0.8)
    valid_size = len(dataset) - train_size

    train_set = dataset
    #train_set, valid_set = random_split(dataset, [train_size, valid_size],
    #                                    generator=torch.Generator().manual_seed(42))

    for i in range(len(train_set)):
        print('=========================')
        print(tokenizer.decode(np.squeeze(train_set[i]['input_ids']), skip_special_tokens=True))
        print(tokenizer.decode(np.squeeze(train_set[i]['labels']), skip_special_tokens=True))
