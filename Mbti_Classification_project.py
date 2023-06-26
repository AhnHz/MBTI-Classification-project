#%%
# 데이터 가져오기

import pandas as pd

# 데이터 불러오기
# 16가지의 class labels을 가진 text dataset
df = pd.read_csv('MBTI_min_1000.csv')

# 데이터 그룹화
df = df.groupby(['type'])

# 그룹별 정렬 함수(람다)
func = lambda g: g.sort_values(by = 'type', ascending=False)[:1000]
df = df.apply(func)

df
# %%
# tokenizer, Embedding

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import numpy as np
from transformers import BertTokenizer

# 토크나이저 설정 : bert-large-uncased 모델
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

labels = {"INFJ" : 0, "INTJ" : 1, "INFP" : 2, "INTP" : 3, "ENFJ" : 4, "ENTJ" : 5,
              "ENFP" : 6, "ENTP" : 7, "ISFJ" : 8, "ISTJ" : 9, "ISFP" : 10, "ISTP" : 11,
                "ESFJ" : 12, "ESTJ" : 13, "ESFP" : 14, "ESTP" : 15}


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        # label 리스트 생성
        self.labels = [labels[label] for label in df['type']]
        # df[posts]의 value를 순차적으로 가져와서 토큰화, padding = 512 설정
        self.texts = [tokenizer(text,
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['posts']]
    def classes(self):
        # 레이블 리스트 반환
        return self.labels

    def __len__(self):
        # 레이블 리스트 길이 반환
        return len(self.labels)

    def get_batch_labels(self, idx):
        # idx에 해당하는 배치 레이블을 numpy 배열로 변환
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # idx에 해당하는 토큰화된 텍스트 배치를 반환
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
# %%
# BERT Fine tuning

# 훈련, 검증, 테스트 데이터 분할
df_train, df_val, df_test = np.split(df.sample(frac=1), [int(.8*len(df)), int(.9*len(df))])
print(len(df_train),len(df_val), len(df_test))
# %%
temp = Dataset(df_test)
print('class 값 : ', temp.labels[0])
print(temp.texts[0])
# %%
from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()
        # 층 생성
        self.bert = BertModel.from_pretrained('bert-large-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(1024, 16)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        
        # 층 적용
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer
# %%
# Train