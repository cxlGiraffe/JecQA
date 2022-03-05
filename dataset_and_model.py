import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from transformers import BartModel,MT5Model,MT5ForConditionalGeneration,BartForSequenceClassification,BartConfig

def pre_answer(data):
    res = []
    res.append(1) if 'A' in data else res.append(0)
    res.append(1) if 'B' in data else res.append(0)
    res.append(1) if 'C' in data else res.append(0)
    res.append(1) if 'D' in data else res.append(0)
    return res

def pre_t5_answer(data):
    res = []
    res.append('对') if 'A' in data else res.append('错')
    res.append('对') if 'B' in data else res.append('错')
    res.append('对') if 'C' in data else res.append('错')
    res.append('对') if 'D' in data else res.append('错')
    return res

class JecT5Dataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

        self.input = []
        self.label = []
        self.en_input,self.en_mask = [],[]
        self.de_input = []
        self.answer = []
        self.id = []

        for i in data:
            option_list = i['option_list']
            for k,v in option_list.items():
                self.input.append('答案:'+v+' 问题:'+i['statement'])
            
            if 'answer' in i:
                la = [j for j in i['answer']]
                lab = pre_t5_answer(la)
                label = lab
            else:
                label=[' ',' ',' ',' ']
            self.label.extend(label)
            for _ in range(4):
                self.answer.append(pre_answer(i['answer']) if 'answer' in i else 0)
                self.id.append(i['id'])
                
        length = len(self.input)
        for i in tqdm(range(0, length, 1000)):
            tokenize_input = self.tokenizer(self.input[i:min(
                i+1000, length)], padding='max_length', max_length=400,truncation=True,return_tensors='pt')
            tokenize_label = self.tokenizer(self.label[i:min(
                i+1000, length)], padding='max_length', max_length=5,truncation=True, return_tensors='pt')
            self.en_input.append(tokenize_input['input_ids'])
            self.en_mask.append(tokenize_input['attention_mask'])
            self.de_input.append(tokenize_label['input_ids'])
            
        self.en_input = torch.cat(self.en_input, dim=0)
        self.en_mask = torch.cat(self.en_mask, dim=0)
        self.de_input = torch.cat(self.de_input, dim=0)

    def __getitem__(self, item):
        return {'input_ids': self.en_input[item], 'attention_mask': self.en_mask[item],'labels':self.de_input[item][1:],'answer':torch.tensor(self.answer[item],dtype=torch.float32),'id':self.id[item]}
            
    def __len__(self):
        return len(self.input)
    
class JecT5CombDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.len = len(data)
        self.input = []
        self.label = []
        self.en_input,self.en_mask = [],[]
        self.de_input = []
        self.answer = []
        self.id = []

        for i in data:
            option_list = i['option_list']
            option_content = [v for k,v in option_list.items()]
            option = [k+v for k,v in option_list.items()]
            op_join = ' '.join(option)
            self.input.append('答案:'+op_join+' 问题:'+i['statement'])
            
            if 'answer' in i:
#                 la = [j+option_list[j] for j in i['answer']]
                la = [j for j in i['answer']]
                lab = pre_t5_answer(la)
#                 label = [lab[j]+option_content[j] for j in range(4)]
                label = ','.join(lab)
            else:
                label=' '
            self.label.append(label)
            
            self.answer.append(pre_answer(i['answer']) if 'answer' in i else 0)
            self.id.append(i['id'])            
            
        for i in tqdm(range(0, self.len, 1000)):
            tokenize_input = self.tokenizer(self.input[i:min(
                i+1000, self.len)], padding='max_length', max_length=400,truncation=True,return_tensors='pt')
            tokenize_label = self.tokenizer(self.label[i:min(
                i+1000, self.len)], padding='max_length', max_length=10,truncation=True, return_tensors='pt')
            self.en_input.append(tokenize_input['input_ids'])
            self.en_mask.append(tokenize_input['attention_mask'])
            self.de_input.append(tokenize_label['input_ids'])
            
        self.en_input = torch.cat(self.en_input, dim=0)
        self.en_mask = torch.cat(self.en_mask, dim=0)
        self.de_input = torch.cat(self.de_input, dim=0)
            
    def __getitem__(self, item):
        return {'input_ids': self.en_input[item], 'attention_mask': self.en_mask[item],'labels':self.de_input[item],'answer':torch.tensor(self.answer[item],dtype=torch.float32),'id':self.id[item]}
            
    def __len__(self):
        return self.len
    
class JecDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.len = len(data)*4
        self.input = []
        self.en_input,self.en_mask = [],[]
        self.answer = []
        self.id = []
        
        for i in data:
            option_list = i['option_list']
            for k,v in option_list.items():
                self.input.append('答案:'+v+' 问题:'+i['statement'])
            self.answer.extend(pre_answer(i['answer']) if 'answer' in i else [0,0,0,0])
            for _ in range(4):
                self.id.append(i['id'])

        for i in tqdm(range(0, self.len, 1000)):
            tokenize_input = self.tokenizer(self.input[i:min(
                i+1000, self.len)], padding='max_length', max_length=500,truncation=True,return_tensors='pt')
            self.en_input.append(tokenize_input['input_ids'])
            self.en_mask.append(tokenize_input['attention_mask'])

        self.en_input = torch.cat(self.en_input, dim=0)
        self.en_mask = torch.cat(self.en_mask, dim=0)
    
    def __getitem__(self, item):
        return {'input_ids': self.en_input[item], 'attention_mask': self.en_mask[item],'answer':torch.tensor(self.answer[item],dtype=torch.long),'id':self.id[item]}
            
    def __len__(self):
        return self.len
    
    
    
# class JecDataset(Dataset):
#     def __init__(self, data, tokenizer):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.len = len(data)
#         self.input = []
#         self.en_input,self.en_mask = [],[]
#         self.answer = []
#         self.id = []
        
#         for i in data:
#             state = i['statement']
#             inp = ','.join(i['option_list'].values()) + state
#             self.answer.append(pre_answer(i['answer']) if 'answer' in i else 0)
#             self.id.append(i['id'])
#             self.input.append(inp)
    
#         for i in tqdm(range(0, self.len, 1000)):
#             tokenize_input = self.tokenizer(self.input[i:min(
#                 i+1000, self.len)], padding='max_length', max_length=500,truncation=True,return_tensors='pt')
#             self.en_input.append(tokenize_input['input_ids'])
#             self.en_mask.append(tokenize_input['attention_mask'])

#         self.en_input = torch.cat(self.en_input, dim=0)
#         self.en_mask = torch.cat(self.en_mask, dim=0)
    
#     def __getitem__(self, item):
#         return {'input_ids': self.en_input[item], 'attention_mask': self.en_mask[item],'answer':torch.tensor(self.answer[item],dtype=torch.float32),'id':self.id[item]}
            
#     def __len__(self):
#         return self.len
    
    


class JecBartModel(nn.Module):
    def __init__(self,model_name):
        super(JecBartModel,self).__init__()
        config = BartConfig.from_pretrained(model_name)
        config.num_labels = 2
        self.bartModel = BartForSequenceClassification.from_pretrained(model_name,config=config)
        
    def forward(self,inputs,labels):
        outputs = self.bartModel(labels=labels,**inputs)
        loss = outputs.loss
        return loss
    
    def generate(self,inputs):
        outputs = self.bartModel(**inputs)
        logits = outputs.logits
        return logits
# hidden_size = 1024
# class JecBartModel1(nn.Module):
#     def __init__(self,model_name):
#         super(JecBartModel1,self).__init__()
#         self.bartModel = BartModel.from_pretrained(model_name)
#         self.dropout = nn.Dropout(0.5)
#         self.linear1 = nn.Linear(hidden_size,4)
        
#     def forward(self,inputs):
#         out = self.bartModel(**inputs).last_hidden_state
#         decoder_last = out[:,-1,:]
#         out1 = torch.sigmoid(self.linear1(self.dropout(decoder_last)))
#         return 0,out1
        

class JecT5Model(nn.Module):
    def __init__(self,model_name):
        super(JecT5Model,self).__init__()
        device_map = {0: [0, 1, 2,3,4],1: [ 5, 6, 7, 8, 9,10,11]}
        self.mt5Model = MT5ForConditionalGeneration.from_pretrained(model_name)
        self.mt5Model.parallelize(device_map)
        
    def forward(self,inputs):
        outputs = self.mt5Model(**inputs)
        loss = outputs.loss
        return loss
    
    def generate(self,inputs):
        outputs = self.mt5Model.generate(**inputs)
        return outputs
# class JecT5Model(nn.Module):
#     def __init__(self,model_name,out1_dim):
#         super(JecT5Model,self).__init__()
#         self.hidden_size = hidden_size
#         self.mt5Model = MT5Model.from_pretrained(model_name)
#         self.dropout = nn.Dropout(0.5)
#         self.linear1 = nn.Linear(self.hidden_size,out1_dim)
        
#     def forward(self,inputs):
#         out = self.mt5Model(**inputs).last_hidden_state
#         decoder_last = out[:,-1,:]
#         out1 = torch.sigmoid(self.linear1(self.dropout(decoder_last)))
    
#         return out1      