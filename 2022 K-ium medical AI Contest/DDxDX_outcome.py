# pandas==1.5.0
# numpy==1.23.4
# sklearn==1.1.2
# scipy==1.9.2;
# torch==1.12.1+cu113
# tensorflow==2.10.0 (with keras)
# transformers==4.23.1

import warnings
warnings.filterwarnings('ignore')

import sys
sout = sys.stdout.write

import pandas as pd
import numpy as np
import random
import time
import datetime

import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, auc, classification_report
from scipy.special import softmax

# GPU 확인하기
n_devices = torch.cuda.device_count()
sout(f'GPU Device count : {n_devices}')

for i in range(n_devices):
    sout(f'GPU Device model : {torch.cuda.get_device_name(i)}')
    
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

# 모델 경로
PATH = './logs/models/model.pt'

# 모델
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
model.cuda()

# 옵티마이저
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # 학습률(learning rate)
                  eps = 1e-8 
                )

batch_size = 8
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']


# csv 로드
sout('\n-----------------------\n\n***** input data! *****\n\n-----------------------\n\n>>>\n')

df = pd.read_csv('./data/' + sys.stdin.readline().rstrip())

sout('\n\n\n\n data_check 5 rows\n\n\n\n')
sout(df.head().to_markdown().strip())
sout('\n\n\n\n')

# 컬럼명수정 - 기존 ['Findings', 'Conclusion\n', 'AcuteInfarction'] to 변경 ['Findings', 'Conclusion', 'AcuteInfarction']
df.columns = 'Findings', 'Conclusions', 'labels'

sout('\n\n\n##### null_check #####\n\n\n')
sout(df.isna().sum().to_markdown())

sout('\n\n\n##### null_fill #####\n\n\n')
# Findings는 결과값이 1인행 223개에 대해서 "Emergency"를 줄인 "ER"을 대체
idx_ER = df[(df.labels == 1) & df.Findings.isna()].index
df.Findings.iloc[idx_ER] = df.Findings.iloc[idx_ER].fillna("ER")

# Findings는 결과값이 0인행 1143개에 대해서 "No Findings"를 줄인 "NF"을 대체
idx_NF = df[(df.labels == 0) & df.Findings.isna()].index
df.Findings.iloc[idx_NF] = df.Findings.iloc[idx_NF].fillna("NF")

# Conclusion 결측치의 경우 1개를 제외하고 Findings가 "MRI for radiosurgery"이므로 결측치를 "GammaKnife"를 줄여서 GK로 대체
df.Conclusions.fillna("GK", inplace = True)

sout('\n\n\n')
sout(df.isna().sum().to_markdown())
sout('\n\n\n')

# 데이터 셔플
test = df.sample(frac=1).reset_index(drop=True)


sout('\n\n\n##### 데이터 셔플 #####\n\n\n')
sout('\t\n\n # 기존 데이타\n\n\n')
sout(df.Findings.head(3).to_markdown())
sout('\t\n\n # 셔플 데이타\n\n\n')
sout(test.Findings.head(3).to_markdown())


sout('\n\n\n ***** 모델 평가 ***** \n\n\n model_evaluation(bert-base-multilingual-cased) Start! \n\n\n')
sout('\n\n batch : ')
sout(str(batch_size))
sout('\n\n optimizer : ')
sout(str(optimizer))

## Bert ##
# Test셋 전처리
sentences = ["[CLS] " + str(s) + " [SEP]" for s in zip(test.Findings, test.Conclusions)]

# 검증 셋 전처리
# [CLS] + 문장 + [SEP]
sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]

# 라벨 데이터
label_list = test['labels'].values

# Word 토크나이저 토큰화
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

# 시퀀스 설정 및 정수 인덱스 변환 & 패딩
MAX_LEN = 128
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# 어텐션 마스크
attention_masks = []
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)
    
# 파이토치 텐서로 변환
test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(label_list)
test_masks = torch.tensor(attention_masks)

# 데이터 설정
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)


# 디바이스 설정
if torch.cuda.is_available():    
    device = torch.device("cuda")
    sout('\n\n There are %d GPU(s) available.' %torch.cuda.device_count())
    sout('\n\n We will use the GPU:')
    sout(str(torch.cuda.get_device_name(0)))
else:
    device = torch.device("cpu")
    sout('\n\n No GPU available, using the CPU instead.')


# 정확도 계산 함수
def flat_accuracy(preds, labels):
    
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)    
    
# 시간 표시 함수
def format_time(elapsed):

    # 반올림
    elapsed_rounded = int(round((elapsed)))
    
    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))

# 확률 계산 함수
def probabilities(proba):
    proba_list = softmax(proba, axis = 1)
    
    return proba_list


#시작 시간 설정
t0 = time.time()

# 평가모드로 변경
model.eval()

# 변수 초기화
pred_list = []
proba_list = []
label_list = []

eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

# 데이터로더에서 배치만큼 반복하여 가져옴
for step, batch in enumerate(test_dataloader):
    # 경과 정보 표시
    if step % 100 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        sout('\n Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

    # 배치를 GPU에 넣음
    batch = tuple(t.to(device) for t in batch)
    
    # 배치에서 데이터 추출
    b_input_ids, b_input_mask, b_labels = batch
    
    # 그래디언트 계산 안함
    with torch.no_grad():     
        # Forward 수행
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
    
    # 로스 구함
    logits = outputs[0]

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
   
    pred_list.append(np.argmax(logits, axis = 1).flatten())
    proba_list.append(probabilities(logits))
    label_list.append(label_ids.flatten())
    
    # 출력 로짓과 라벨을 비교하여 정확도 계산
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1
    
y_pred = np.concatenate(pred_list)
y_proba = np.concatenate(proba_list)
y_true = np.concatenate(label_list)  
    
sout("\n Accuracy: {0:.6f}".format(eval_accuracy/nb_eval_steps))
sout("\n Test took: {:}".format(format_time(time.time() - t0)))
sout('\n\n')
sout('Done!\n\n')


# 데이터 확인 확인(예측값, 정답값, 확률)
sout(f' Pred : {y_pred.shape} , Labels : {y_true.shape} , Proba : {y_proba.shape}\n')



# 성능 지표 확인
sout('Result!\n')
sout(classification_report(y_true, y_pred))

fpr, tpr, thresh = roc_curve(y_true, y_proba[:,1])

roc_auc = roc_auc_score(y_true, y_proba[:,1])

sout('\n\n AUROC = %0.6f'%roc_auc)