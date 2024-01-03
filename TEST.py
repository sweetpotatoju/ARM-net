import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold
import category_encoders as ce
import sklearn
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error

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
df = pd.read_csv(csv_file_path)
# print(df)

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

    params = {
        'depth': 8,
        'n_estimators': 2048,

        'learning_rate_weights': 0.005,
        'learning_rate_index': 0.01,
        'learning_rate_values': 0.01,
        'learning_rate_leaf': 0.01,

        'optimizer': 'SWA',
        'cosine_decay_steps': 0,

        'initializer': 'RandomNormal',

        'loss': 'crossentropy',
        'focal_loss': False,

        'from_logits': True,
        'apply_class_balancing': True,

        'dropout': 0.0,

        'selected_variables': 0.8,
        'data_subset_fraction': 1.0,
    }

    args = {
        'epochs': 500,
        'early_stopping_epochs': 25,
        'batch_size': 64,

        'objective': 'binary',
        'evaluation_metrics': ['F1'],  # F1, Accuracy, R2
        'random_seed': 42,
        'verbose': 1,
    }

    model = GRANDE(params=params, args=args)

    model.fit(X_train= X_fold_train_resampled,
          y_train=y_fold_train_resampled,
          X_val=X_valid,
          y_val=y_valid)


    preds = model.predict(X_test_Nomalized)

    threshold = 0.5  # 임계값 설정

    binary_preds = [1 if prob >= threshold else 0 for prob in preds[:, 1]]  # preds[:, 1]는 1로 예측될 확률을 나타냄
    binary_preds = np.array(binary_preds).reshape(-1, 1)  # NumPy 배열로 변환하고 열의 차원을 1로 지정
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