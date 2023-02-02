# Team.DDxDX

## 인터프리터
    jupyterlab==3.4.8

## 라이브러리 버젼
    python==3.10.8 (warnings, random, time, datetime, re)
    pandas==1.5.0
    numpy==1.23.4
    sklearn==1.1.2
    scipy==1.9.2
    torch==1.12.1 + cu113
    tensorflow==2.10.0 (keras)
    transformers==4.23.1
    matplotlib==3.6.1 (roc_curve plot)


### 모델경로
    `/logs/models/model.pt`

## Dataset 경로
    `/data/`

## ☆실행☆
    1. 지표측정 : `python DDxDX_outcome.py` > 파일명입력 (ex. ValidationSet.csv)
    2. 확률값산출 : 'python DDxDX_proba.py' > 파일명입력 (ex. ValidationSet.csv)

## 결과출력
    1. Accuracy
    2. classification_report
    3. AUROC
    4. proba

## 모델 관련 스크립트
# 작업물을 별도 html파일로 저장하여 첨부 하였습니다.
    1. DDxDX_final.html - 모델 학습
    2. DDxDX_singleword - 사용자 문장 입력
    3. DDxDX_val - 테스트셋 평가