from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore") #경고 안보이게

#데이터를 불러 옵니다.
df = pd.read_csv("./data/house_train.csv")

pd.set_option('display.max_row', None)
#pd.set_option('display.max_columns', None)  # 판다스 열을 무제한 확장
#pd.set_option('display.width', 1000) #디스플레이 폭 확장
#데이터를 미리 살펴 보겠습니다.
#print(df.head(10))
#데이터가 어떤 유형으로 이루어져 있는지 알아봅니다.
print(df.dtypes)
print('========================================================')
#속성별로 결측치(데이터에서 누락된 부분)가 몇 개인지 확인합니다.
print(df.isnull().sum().sort_values(ascending=False))
print(df.first_valid_index()) #last_valid_index마지막으로 결측치가 아닌값이 나오는 행의 인덱스를 출력합니다.

#카테고리형 변수를 0과 1로 이루어진 변수로 바꾸어 줍니다.(12장 3절)
df = pd.get_dummies(df)
df.to_csv("./data/house_train-1.csv")
#결측치를 전체 칼럼의 평균으로 대체하여 채워줍니다.
df = df.fillna(df.mean(axis=1)) #axis=1 행렬 지정
pd.set_option('display.max_columns', None)
df.to_csv("./data/house_train-2.csv")
#업데이트된 데이터프레임을 출력해 봅니다.
#print(df.head(10))

#데이터 사이의 상관 관계를 저장합니다.
df_corr=df.corr() #Correlation
df_corr.to_csv("./data/house_train-corr.csv")

#집 값과 관련이 큰 것부터 순서대로 저장합니다.
df_corr_sort=df_corr.sort_values('SalePrice', ascending=False)
df_corr_sort=df_corr.sort_values(['SalePrice','FullBath'], ascending=[False,True])
df_corr_sort.to_csv("./data/house_train-sort.csv")

#print(df_corr.head(10))
#집 값과 관련도가 가장 큰 10개의 속성들을 출력합니다.
print(df_corr_sort['SalePrice'].head(10))

#집 값과 관련도가 가장 높은 속성들을 추출해서 상관도 그래프를 그려봅니다.
cols=['SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']
sns.pairplot(df[cols])
plt.show();

#집 값을 제외한 나머지 열을 저장합니다.
cols_train=['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']
X_train_pre = df[cols_train]
print(X_train_pre)

#집 값을 저장합니다.
y = df['SalePrice'].values

#전체의 80%를 학습셋으로, 20%를 테스트셋으로 지정합니다.
X_train, X_test, y_train, y_test = train_test_split(X_train_pre, y, test_size=0.2)
#모델의 구조를 설정합니다.
print(X_train.shape)
print('X_train.shape[1]===================>',X_train.shape[1])
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))
model.summary()

#모델을 실행합니다.
model.compile(optimizer ='adam', loss = 'mean_squared_error')

# 20회 이상 결과가 향상되지 않으면 자동으로 중단되게끔 합니다.
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

# 모델의 이름을 정합니다.
modelpath="./data/model/Ch15-house.hdf5"

# 최적화 모델을 업데이트하고 저장합니다.
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)

#실행 관련 설정을 하는 부분입니다. 전체의 20%를 검증셋으로 설정합니다.
history = model.fit(X_train, y_train, validation_split=0.25, epochs=2000, batch_size=32, callbacks=[early_stopping_callback, checkpointer])

# 예측 값과 실제 값, 실행 번호가 들어갈 빈 리스트를 만듭니다.
real_prices =[]
pred_prices = []
X_num = []

print('X_test===>',X_test.shape)
print(X_test.dtypes)
#print(X_test)
# 25개의 샘플을 뽑아 실제 값, 예측 값을 출력해 봅니다.
n_iter = 0
print(model.predict(X_test))
Y_prediction = model.predict(X_test).flatten() # flatten() 함수는 데이터 배열이 몇 차원이든 모두 1차원으로 바꿔 읽기 쉽게 해주는 함수
print("Y_prediction==========>",Y_prediction)
Y_prediction=Y_prediction.flatten()
for i in range(25):
#    print('X_test===>',X_test[i])
    real = y_test[i]
    prediction = Y_prediction[i]
    print("실제가격: {:.2f}, 예상가격: {:.2f}".format(real, prediction))
    real_prices.append(real)
    pred_prices.append(prediction)
    n_iter = n_iter + 1
    X_num.append(n_iter)

#그래프를 통해 샘플로 뽑은 25개의 값을 비교해 봅니다.
print(pred_prices)

plt.plot(X_num, pred_prices, label='predicted price')
plt.plot(X_num, real_prices, label='real price')
plt.legend()
plt.show()
