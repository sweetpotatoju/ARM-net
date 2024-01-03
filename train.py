import csv
import os
import time
import argparse

import torch
import tqdm as tqdm
import zero as zero
from scipy.sparse import csr_matrix
from torch import nn, FloatTensor, device
import torch.backends.cudnn as cudnn
from torch import optim
from torch.nn.modules import loss
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset

from models.model_utils import create_model

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold
import category_encoders as ce
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error

from models.armnet_1h import ARMNetModel as ARMNet1H

pd_list = []
pf_list = []
bal_list = []
fir_list = []

def classifier_eval(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print('혼동행렬 : ', cm)
    PD = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    print('PD : ', PD)
    PF = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    print('PF : ', PF)
    balance = 1 - (((0 - PF) * (0 - PF) + (1 - PD) * (1 - PD)) / 2)
    print('balance : ', balance)
    FI = (cm[1, 1] + cm[0, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    FIR = (PD - FI) / PD
    print('FIR : ', FIR)

    return PD, PF, balance, FIR

# CSV 파일 경로를 지정
csv_file_path ="JDT.csv"

# CSV 파일을 데이터프레임으로 읽어오기
# CSV 파일을 데이터프레임으로 읽어오기 (첫 번째 행을 제외)
df = pd.read_csv(csv_file_path)

# 데이터프레임에서 특징(X)과 목표 변수(y) 추출
X = df.drop(columns=['class'])
y = df['class']  # 'class' 열을 목표 변수로 사용


# 데이터프레임에서 특징(X)과 목표 변수(y) 추출
X = df.drop(columns=['class'])
y = df['class']  # 'target' 열을 목표 변수로 사용

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# K-겹 교차 검증을 설정합니다
k = 10 # K 값 (원하는 폴드 수) 설정
kf = KFold(n_splits=k, shuffle=True, random_state=42)

scaler = MinMaxScaler()
X_test_Nomalized =scaler.fit_transform(X_test)

# K-겹 교차 검증 수행
for train_index, val_index in kf.split(X_train):
    X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

#전처리
    # Min-Max 정규화 수행(o)
    X_fold_train_normalized = scaler.fit_transform(X_fold_train)
    X_fold_val_normalized = scaler.transform(X_fold_val)

    # SMOTE를 사용하여 학습 데이터 오버샘플링
    smote = SMOTE(random_state=42)
    X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(X_fold_train_normalized, y_fold_train)

    X_fold_train_resampled_tensor = torch.FloatTensor(X_fold_train_resampled)
    y_fold_train_resampled_tensor = torch.FloatTensor(y_fold_train_resampled)

    X_valid_tensor = torch.FloatTensor(X_valid)
    y_valid_tnesor = torch.FloatTensor(y_valid)

    # DataLoader를 사용하여 데이터로더 생성
    batch_size = 32
    train_dataset = TensorDataset(X_fold_train_resampled_tensor, y_fold_train_resampled_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tnesor)

def get_args():
    parser = argparse.ArgumentParser(description='ARMOR framework')
    parser.add_argument('--exp_name', default='test', type=str, help='exp name for log & checkpoint')
    # model config
    parser.add_argument('--model', default='armnet', type=str, help='model type, afn, arm etc')
    parser.add_argument('--nfeat', type=int, default=61, help='the number of features')
    parser.add_argument('--nfield', type=int, default=10, help='the number of fields')
    parser.add_argument('--nemb', type=int, default=10, help='embedding size')
    parser.add_argument('--k', type=int, default=3, help='interaction order for hofm/dcn/cin/gcn/gat/xdfm')
    parser.add_argument('--h', type=int, default=600, help='afm/cin/afn/armnet/gcn/gat hidden features/neurons')
    parser.add_argument('--mlp_nlayer', type=int, default=2, help='the number of mlp layers')
    parser.add_argument('--mlp_nhid', type=int, default=300, help='mlp hidden units')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--nattn_head', type=int, default=4, help='the number of attention heads, gat/armnet')
    # for AFN/ARMNet
    parser.add_argument('--ensemble', action='store_true', default=False, help='to ensemble with DNNs')
    parser.add_argument('--dnn_nlayer', type=int, default=2, help='the number of mlp layers')
    parser.add_argument('--dnn_nhid', type=int, default=300, help='mlp hidden units')
    parser.add_argument('--alpha', default=1.7, type=float, help='entmax alpha to control sparsity')
    # optimizer
    parser.add_argument('--epoch', type=int, default=100, help='number of maximum epochs')
    parser.add_argument('--patience', type=int, default=1, help='number of epochs for stopping training')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--lr', default=0.003, type=float, help='learning rate, default 3e-3')
    parser.add_argument('--eval_freq', type=int, default=10000, help='max number of batches to train per epoch')

    args = parser.parse_args()
    return args

args = get_args()

 #모델 인스턴스 생성
model = ARMNet1H(args.nfeat, args.nfield, args.nemb, args.alpha, args.h, args.nemb, args.mlp_nlayer, args.mlp_nhid, args.dropout, args.ensemble, args.dnn_nlayer, args.dnn_nhid)
# 입력 데이터 준비 (예: 훈련 데이터)
# 텐서로 변환
input_data = {'value': torch.FloatTensor(X_fold_train_resampled), 'label': torch.FloatTensor(y_fold_train_resampled)}

# print("Tensor 'value':", input_data['value'])
# print("Tensor 'label':", input_data['label'])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
opt_metric = nn.BCEWithLogitsLoss(reduction='mean')

# GPU 사용 가능한 경우 모델을 GPU로 이동
if torch.cuda.is_available():
    model = model.cuda()

cudnn.benchmark = True

# patience_cnt = 0
# batch_size = 256
# train_loader = zero.data.IndexLoader(len(X_train), batch_size, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# progress = zero.ProgressTracker(patience=100)

# 학습
n_epochs = 100
report_frequency = max(len(train_loader) // (batch_size * 5), 1)  # 0으로 나누는 것을 방지하기 위해 최소값을 1로 설정
for epoch in range(1, n_epochs + 1):
    for iteration, (x_batch, y_batch) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        predictions = model({'value': x_batch})
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        if iteration % report_frequency == 0:
            print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f}')


# def run(epoch, model, data_loader, optimizer, namespace='train'):
    #     if namespace == 'train':
    #         model.train()
    #     else:
    #         model.eval()
    #
    #     for batch_idx, batch in enumerate(data_loader):
    #         # 'y'의 데이터 유형을 변환
    #         target = torch.tensor(batch['y'], dtype=torch.long)  # 예: float32로 변환
    #         # 또는
    #         # target = torch.tensor(batch['y'], dtype=torch.long)  # 예: long으로 변환
    #         # 또는 다른 적절한 데이터 유형으로 변환
    #         # 변환 종료
    #
    #         if torch.cuda.is_available():
    #             target = target.cuda(non_blocking=True)
    #
    #         if namespace == 'train':
    #             optimizer.zero_grad()
    #             output = model(batch)
    #             loss = opt_metric(output, target)
    #             loss.backward()
    #             optimizer.step()
    #         else:
    #             with torch.no_grad():
    #                 output = model(batch)
    #                 loss = opt_metric(output, target)
    #
    #         # 필요한 경우에 따라 손실을 출력하거나 로그에 기록
    #         if batch_idx % args.log_interval == 0:
    #             print(f'{namespace} Epoch: {epoch} [{batch_idx}/{len(data_loader)}] Loss: {loss.item():.6f}')
    #
    #
    # # 훈련 루프
    # for epoch in range(args.epoch):
    #     run(epoch, model, data_loader, optimizer, namespace='train')

# # 테스트 데이터에 대한 예측 수행
# test_input_data = {'id': torch.FloatTensor(X_valid), 'value': torch.LongTensor(y_valid)}
# with torch.no_grad():
#     test_output = model(test_input_data)
#
# # 예측 결과 확인
# print(test_output)

# def main():
#     model = create_model(args)
#     print("model")
#     # optimizer
#     opt_metric = nn.BCEWithLogitsLoss(reduction='mean')
#     if torch.cuda.is_available(): opt_metric = opt_metric.cuda()
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
#     # gradient clipping
#     for p in model.parameters():
#         p.register_hook(lambda grad: torch.clamp(grad, -1., 1.))
#     cudnn.benchmark = True
#
#     train_loader = X_fold_train_resampled, y_fold_train_resampled
#     val_loader = X_valid, y_valid
#     test_loader =X_test_Nomalized
#     patience_cnt = 0
#     for epoch in range(args.epoch):
#
#         run(epoch, model, train_loader, opt_metric, optimizer=optimizer)
#         run(epoch, model, val_loader, opt_metric)
#         run(epoch, model, test_loader, opt_metric)
#
#
# def run(epoch, model, data_loader, opt_metric, optimizer=None, namespace='train'):
#     if optimizer:
#         model.train()
#         print("f")
#     else: model.eval()
#
#
#     for batch_idx, batch in enumerate(data_loader):
#         target = batch['y']
#         print(batch)
#         if torch.cuda.is_available():
#             batch['id'] = batch['id'].cuda(non_blocking=True)
#             batch['value'] = batch['value'].cuda(non_blocking=True)
#             target = target.cuda(non_blocking=True)
#
#         if namespace == 'train':
#             y = model(batch)
#             loss = opt_metric(y, target)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         else:
#             with torch.no_grad():
#                 y = model(batch)
#                 loss = opt_metric(y, target)
#
#
#
# #     threshold = 0.5  # 임계값 설정
# #
# #     binary_preds = [1 if prob >= threshold else 0 for prob in preds[:, 1]]  # preds[:, 1]는 1로 예측될 확률을 나타냄
# #     binary_preds = np.array(binary_preds).reshape(-1, 1)  # NumPy 배열로 변환하고 열의 차원을 1로 지정
# #     y_test = np.array(y_test).reshape(-1, 1)
# #     PD, PF, balance, FIR = classifier_eval(y_test, binary_preds)
# #     pd_list.append(PD)
# #     pf_list.append(PF)
# #     bal_list.append(balance)
# #     fir_list.append(FIR)
# #
# # print('avg_PD: {}'.format((sum(pd_list) / len(pd_list))))
# # print('avg_PF: {}'.format((sum(pf_list) / len(pf_list))))
# # print('avg_balance: {}'.format((sum(bal_list) / len(bal_list))))
# # print('avg_FIR: {}'.format((sum(fir_list) / len(fir_list))))


