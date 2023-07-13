import torch
import numpy as np

class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, tokenizer):
        self.sentences = sentences
        self.nums = len(sentences)
        self.tokenizer = tokenizer
        self.max_length = 128

    def __getitem__(self, idx):
        random_num = np.random.randint(1, 2)
        combined_sentences = self.sentences[idx:(idx+random_num)]
        combined_sentences = ''.join([i['sentence'] for i in combined_sentences])
        combined_sentences = combined_sentences[:self.max_length]
        percent = np.random.randint(4, 6) * 0.1
        ch_pt = int(len(combined_sentences) * percent)
        
        input_text  = combined_sentences[:ch_pt]
        output_text = combined_sentences[ch_pt:]

        #inputs = self.preprocess_seq2seq(input_text, output_text)
        inputs = self.preprocess_casual(input_text, output_text)

        return inputs

    def __len__(self):
        return self.nums

    def preprocess_seq2seq(self, input_text, output_text):
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

    def preprocess_casual(self, input_text, output_text):
        model_inputs = self.tokenizer(input_text)#, max_length=self.max_length,
                               #padding="max_length", truncation=True)

        labels = self.tokenizer(output_text)#, max_length=self.max_length,
                                #padding="max_length", truncation=True)

        sample_input_ids = model_inputs["input_ids"]
        label_input_ids = labels["input_ids"] + [self.tokenizer.pad_token_id]
        #print(len(sample_input_ids), len(label_input_ids))
        model_inputs["input_ids"] = sample_input_ids + label_input_ids
        labels["input_ids"] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"] = [1] * len(model_inputs["input_ids"])
        #print(model_inputs)

        sample_input_ids = model_inputs["input_ids"]
        label_input_ids = labels["input_ids"]
        model_inputs["input_ids"] = [self.tokenizer.pad_token_id] * (self.max_length - len(sample_input_ids)) + sample_input_ids
        model_inputs["attention_mask"] = [0] * (self.max_length - len(sample_input_ids)) + model_inputs["attention_mask"]
        labels["input_ids"] = [-100] * (self.max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"][:self.max_length])
        model_inputs["attention_mask"] = torch.tensor(model_inputs["attention_mask"][:self.max_length])
        labels["input_ids"] = torch.tensor(labels["input_ids"][:self.max_length])

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

if __name__ == '__main__':
    from transformers import AutoTokenizer
    import torch
    import re
    from torch.utils.data import DataLoader, random_split
    import json

    def write_jsonl(filename, data_set):
        out_list = []
        for i in range(len(data_set)):

            label_ids  = data_set[i]['labels'][0]
            label_ids  = np.array(label_ids)
            label_ids[label_ids == -100] = 0

            input_text = tokenizer.decode(data_set[i]['input_ids'][0], skip_special_tokens=True)
            label_text = tokenizer.decode(label_ids, skip_special_tokens=True)
            out_list.append({'prompt': input_text, 'completion': label_text})

        with open(f'{filename}.jsonl', 'w', encoding="utf8") as f:
            for row in out_list:
                json.dump(row, f, ensure_ascii=False)
                f.write('\n')
        

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

    dataset = SentenceDataset(data[:8000], tokenizer)

    train_size = int(len(dataset) * 0.8)
    valid_size = len(dataset) - train_size

    train_set, valid_set = random_split(dataset, [train_size, valid_size],
                                        generator=torch.Generator().manual_seed(42))

    #write_jsonl('train', train_set)
    #write_jsonl('valid', valid_set)

    for i in range(1):
        ins = train_set[5]['input_ids']
        outs = train_set[5]['labels']
        outs[outs == -100] = 0
        input_text = tokenizer.decode(np.squeeze(ins), skip_special_tokens=True)
        output_text = tokenizer.decode(np.squeeze(outs), skip_special_tokens=True)
        print(input_text)
        print('########')
        print(output_text)

        break
