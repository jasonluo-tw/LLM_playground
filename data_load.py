import torch
import numpy as np

class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, tokenizer):
        self.sentences = sentences
        self.nums = len(sentences)
        self.tokenizer = tokenizer
        self.max_length = 256

    def __getitem__(self, idx):
        random_num = np.random.randint(1, 4)
        combined_sentences = self.sentences[idx:(idx+random_num)]
        combined_sentences = ''.join([i['sentence'] for i in combined_sentences])
        percent = np.random.randint(4, 6) * 0.1
        ch_pt = int(len(combined_sentences) * percent)
        
        input_text  = combined_sentences[:ch_pt]
        output_text = combined_sentences[ch_pt:]

        inputs = self.tokenizer(input_text, max_length=self.max_length, 
                                padding="max_length", truncation=True, 
                                return_tensors="pt")

        labels = self.tokenizer(output_text, max_length=self.max_length,
                                padding="max_length", truncation=True,
                                return_tensors="pt") 

        labels = labels["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels

        return inputs

    def __len__(self):
        return self.nums

if __name__ == '__main__':
    from transformers import AutoTokenizer
    import torch
    import re

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

    train_loader = torch.utils.data.DataLoader(
         dataset=dataset,
         batch_size=1,
         num_workers=1,
         pin_memory=True,
         shuffle=True)

    for i in train_loader:
        ins = i[0]['input_ids']
        outs = i[0]['labels']
        outs[outs == -100] = 0
        input_text = tokenizer.decode(np.squeeze(ins), skip_special_tokens=True)
        output_text = tokenizer.decode(np.squeeze(outs), skip_special_tokens=True)
        print(input_text)
        print(output_text)

        break
