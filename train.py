import argparse

import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

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
        #  for AFN/ARMNet
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

# CSV 파일 경로를 지정
csv_file_path ="JDT.csv"

# CSV 파일을 데이터프레임으로 읽어오기
# CSV 파일을 데이터프레임으로 읽어오기 (첫 번째 행을 제외)
df = pd.read_csv(csv_file_path)

# 데이터프레임에서 특징(X)과 목표 변수(y) 추출
X = df.drop(columns=['class'])
y = df['class']  # 'class' 열을 목표 변수로 사용


X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)


scaler = MinMaxScaler()
X_test_Nomalized =scaler.fit_transform(X_test)
X_test_tensor = torch.FloatTensor(X_test_Nomalized)

# K-겹 교차 검증을 설정합니다
k = 10 # K 값 (원하는 폴드 수) 설정
kf = KFold(n_splits=k, shuffle=True, random_state=42)

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


    # DataLoader를 사용하여 데이터로더 생성
    batch_size = 64
    train_dataset = TensorDataset(X_fold_train_resampled_tensor, y_fold_train_resampled_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # valid_dataset = TensorDataset(X_valid_tensor, y_valid_tnesor)

    args = get_args()

    #모델 인스턴스 생성
    model = ARMNet1H(args.nfeat, args.nfield, args.nemb, args.alpha, args.h, args.nemb, args.mlp_nlayer, args.mlp_nhid, args.dropout, args.ensemble, args.dnn_nlayer, args.dnn_nhid)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    opt_metric = nn.BCEWithLogitsLoss(reduction='mean')

    # GPU 사용 가능한 경우 모델을 GPU로 이동
    if torch.cuda.is_available():
        model = model.cuda()

    cudnn.benchmark = True

    # 학습
    n_epochs = 10
    report_frequency = max(len(train_loader) // (batch_size * 5), 1)  # 0으로 나누는 것을 방지하기 위해 최소값을 1로 설정
    for epoch in range(1, n_epochs + 1):
        # tqdm을 사용하여 진행 상황을 시각적으로 표시
        for iteration, (x_batch, y_batch) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            model.train()
            optimizer.zero_grad()
            predictions = model({'value': x_batch})

            # 출력 레이어에 sigmoid 활성화 함수 추가
            predictions = torch.sigmoid(predictions)

            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            if iteration % report_frequency == 0:
                # tqdm에서는 진행 상황을 자동으로 표시하므로 이전에 있던 print문은 제거해도 됩니다.
                pass

    # 모델 평가 모드로 전환
    model.eval()
    # 새로운 데이터에 대한 예측 수행
    with torch.no_grad():
        new_data = torch.randn((5, 10))  # 새로운 입력 데이터
        predictions = model({'value': X_test_tensor})
        predictions = torch.sigmoid(predictions)  # 출력 레이어에 sigmoid 활성화 함수 추가

    threshold = 0.5  # 임계값 설정

    binary_predictions = (predictions > threshold).int()

    # 이진 분류 결과 출력
    print("Binary Predictions:", binary_predictions)

    # NumPy 배열로 변환하고 열의 차원을 1로 지정
    binary_preds = np.array(binary_predictions).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    PD, PF, balance, FIR = classifier_eval(y_test, binary_preds)
    pd_list.append(PD)
    pf_list.append(PF)
    bal_list.append(balance)
    fir_list.append(FIR)


print('avg_PD: {}'.format((sum(pd_list) / len(pd_list))))
print('avg_PF: {}'.format((sum(pf_list) / len(pf_list))))
print('avg_balance: {}'.format((sum(bal_list) / len(bal_list))))
print('avg_FIR: {}'.format((sum(fir_list) / len(fir_list))))


