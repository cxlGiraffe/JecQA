#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import os
import warnings
import json
import torch.nn.functional as F
import torch.optim as optim
import configparser
from tqdm import tqdm
from train_tools import EarlyStopping,get_logger
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score
from transformers import AutoTokenizer,BertTokenizer,BartModel,BartConfig,Adafactor
from torch.utils.data.dataloader import DataLoader
from data_prepare import data_prepare,process_answer
from dataset_and_model import JecDataset,JecBartModel1


# In[2]:


config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")
section_name = 'bart2'

batch_size = config.getint(section_name,'batch_size')
epochs = config.getint(section_name,'epochs')
device = torch.device('cuda:'+config.get(section_name,'gpu') if torch.cuda.is_available else 'cpu')
lr = config.getfloat(section_name,'batch_size')
accum_step = config.getint(section_name,'accum_step')
patience = config.getint(section_name,'patience')
model_name = config.get(section_name,'model_name')
model_savepath = config.get(section_name,'model_savepath')
predict_savepath = config.get(section_name,'predict_savepath')
log_path = config.get(section_name,'log_path')
evaluate_path = config.get(section_name,'evaluate_path')

train_datapath = config.get('DEFAULT','train_datapath')
valid_datapath = config.get('DEFAULT','valid_datapath')
test_datapath = config.get('DEFAULT','test_datapath')


# In[3]:


train_data,valid_data,test_data = data_prepare(train_datapath,test_datapath)


# In[4]:


tokenizer = BertTokenizer.from_pretrained(model_name)


# In[5]:


train_data = JecDataset(train_data,tokenizer)
valid_data = JecDataset(valid_data,tokenizer)
train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size)


# In[6]:


net = JecBartModel1(model_name)


# In[7]:


def get_metric(target,pred):
    acc = accuracy_score(target,pred)
    f1 = f1_score(target,pred,average='macro')
    precision = precision_score(target,pred,average='macro')
    recall = recall_score(target,pred,average='macro')
    return acc,f1,precision,recall

def train_per_epoch(train_loader,device,model,loss,opt,accum_step,log_per_step=None):
    l_sum = 0
    for i,batch in enumerate(tqdm(train_loader)):
        target = batch['answer'].to(device)
        inputs = {key:value.to(device) for key,value in batch.items() if key!='answer' and key!='id'}
        inputs['decoder_input_ids'] = inputs['input_ids']
        inputs['decoder_attention_mask'] = inputs['attention_mask']
        _,out = model(inputs)
        l = loss(out,target)
        l_sum += l.item()
        l = l/accum_step
        l.backward()
        if (i+1)%accum_step==0 or (i+1)==len(train_loader):
            opt.step()
            opt.zero_grad()

    return l_sum/len(train_loader)

def evaluate_per_epoch(valid_loader,device,model,loss,epoch):
    predicts = []
    targets = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            target = batch['answer'].to(device)
            inputs = {key:value.to(device) for key,value in batch.items() if key!='answer' and key!='id'}
            inputs['decoder_input_ids'] = inputs['input_ids']
            inputs['decoder_attention_mask'] = inputs['attention_mask']
            _,out = model(inputs)
            targets.append(target.cpu())
            predicts.append(out.cpu())
            
        targets = torch.cat(targets,dim=0)
        out = torch.cat(predicts,dim=0)
        final_out = torch.where(out>=0.5,1,0)
        valid_acc ,valid_f1 ,valid_precision ,valid_recall = get_metric(targets,final_out)
    
    with open(evaluate_path,'a') as f:
        f.write('epoch:'+str(epoch)+'\n')
        for i in final_out[:100]:
            f.write(str(i)+'\n')
    return valid_acc,valid_precision,valid_recall,valid_f1
    
def train(train_loader,valid_loader,device,model,epochs,lr,accum_step,logger,log_per_step=None,early_stop=None):
    device = device
    logger.info(f'train on {device}')
    model = model.to(device)
    
    loss = nn.BCELoss()
    opt = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    for epoch in range(epochs):
        model.train()
        train_loss = train_per_epoch(train_loader,device,model,loss,opt,accum_step)
        logger.info(f'epoch:{epoch},loss:{train_loss:.5f}')
        
        model.eval()
        warnings.filterwarnings('ignore')
        valid_acc,valid_precision,valid_recall,valid_f1 = evaluate_per_epoch(valid_loader,device,model,loss,epoch)
        logger.info(f'epoch:{epoch},valid_acc:{valid_acc:.5f},valid_precision:{valid_precision:.5f},valid_recall:{valid_recall:.5f},valid_f1:{valid_f1:.5f}')
        
        if early_stop is not None:
            stop = early_stop(valid_acc,model)
            if stop:
                logger.info('Early stopping')
                break


# In[8]:


early_stopping = EarlyStopping(model_savepath,patience)
torch.cuda.empty_cache()
if os.path.exists(log_path):
    os.remove(log_path)
if os.path.exists(evaluate_path):
    os.remove(evaluate_path)
logger = get_logger(log_path)
train(train_loader,valid_loader,device,net,epochs,lr,accum_step,logger,early_stop=early_stopping)


# In[9]:


test_data = JecDataset(test_data,tokenizer)
test_loader = DataLoader(test_data, batch_size=batch_size)
net.load_state_dict(torch.load(model_savepath))  


# In[10]:


def test(test_loader,device,model):
    predicts = []
    all_ids = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            ids = batch['id']
            inputs = {key:value.to(device) for key,value in batch.items() if key!='answer' and key!='id'}
            inputs['decoder_input_ids'] = inputs['input_ids']
            inputs['decoder_attention_mask'] = inputs['attention_mask']
            _,out = model(inputs)
            predicts.append(out.cpu())
            all_ids.extend(ids)
            
        out = torch.cat(predicts,dim=0)
        final_out = torch.where(out>=0.5,1,0)
    return final_out,all_ids


# In[11]:


all_res,all_ids = test(test_loader,device,net)


# In[12]:


all_res = all_res.tolist()
alpha_ans = [process_answer(i) for i in all_res]
alpha_ans = [op if len(op)>0 else ['C'] for op in alpha_ans]
myres = dict(zip(all_ids,alpha_ans))       


# In[13]:


with open(predict_savepath,'w') as f:
    json.dump(myres,f,ensure_ascii=False,indent=4)

