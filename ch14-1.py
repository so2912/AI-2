from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint

# 데이터를 입력합니다.
df = pd.read_csv('./data/wine.csv', header=None)

pd.set_option('display.max_row', None)
pd.set_option('display.max_columns', None)  # 판다스 열을 무제한 확장
pd.set_option('display.width', 1000) #디스플레이 폭 확장

# 데이터를 미리 보겠습니다.
print(df)

# 와인의 속성을 X로 와인의 분류를 y로 저장합니다.
X = df.iloc[:,0:12]
y = df.iloc[:,12]

#학습셋과 테스트셋으로 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# 모델 구조를 설정합니다.
model = Sequential()
model.add(Dense(30,  input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 저장의 조건을 설정합니다.
modelpath="./data/model/all/bestModel.hdf5"

from keras.callbacks import ModelCheckpoint, EarlyStopping
early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=5)
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_accuracy', save_best_only=True, verbose=1)

# 모델을 실행합니다.
his=model.fit(X_train, y_train, epochs=500, batch_size=500, validation_split=0.25, callbacks=[early_stopping_callback,checkpointer],verbose=0) # 0.8 x 0.25 = 0.2
# history에 저장된 학습 결과를 확인해 보겠습니다.
#print(his.history)
# 테스트 결과를 출력합니다.
score=model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])

hist_df=pd.DataFrame(his.history) #판다스 배열로 전환
print(hist_df)
# y_vloss에 테스트셋(여기서는 검증셋)의 오차를 저장합니다.
y_vloss=hist_df['val_loss']
y_loss=hist_df['loss']
# x 값을 지정하고 테스트셋(검증셋)의 오차를 빨간색으로, 학습셋의 오차를 파란색으로 표시합니다.
x_len = np.arange(len(y_loss))

import matplotlib.pyplot as plt
plt.plot(x_len, y_vloss, "o", c="red", markersize=2, label='Testset_loss')
plt.plot(x_len, y_loss, "o", c="blue", markersize=2, label='Trainset_loss')
plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
