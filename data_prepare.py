import json
from sklearn.model_selection import train_test_split

def data_prepare(train_datapath,test_datapath):
    raw_data = []
    with open(train_datapath,'r') as f:
        for i in f.readlines():
            raw_data.append(json.loads(i))

    train_data, valid_data = train_test_split(raw_data, test_size=1000, shuffle=True)
    test_data = []
    with open(test_datapath,'r') as f:
        for i in f.readlines():
            test_data.append(json.loads(i))
            
    return train_data,valid_data,test_data

def process_answer(data):
    res = []
    if data[0]:
        res.append('A')
    if data[1]:
        res.append('B')
    if data[2]:
        res.append('C')
    if data[3]:
        res.append('D')
    return res