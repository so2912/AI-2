import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 아이리스 데이터를 불러옵니다
df = pd.read_csv('./data/iris3.csv')
# 첫 5줄을 봅니다
print(df.head())

#sns.pairplot(df, hue='species',palette="husl", vars=['sepal_length', 'petal_length'])
#plt.show()

X = df.iloc[:,0:4]
y = df.iloc[:,4]
print(y)
print(X[0:5])
print(y[0:5])

# 원-핫 인코딩 처리를 합니다.
yy = pd.get_dummies(y)
#yy.replace((True,False) , (1,0) , inplace = True) #Bool값을 정수로
print(yy)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(12,  input_dim=4, activation='relu'))
model.add(Dense(8,  activation='relu'))
model.add(Dense(3, activation='softmax')) #가장 큰 값의 위치

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(X, yy, epochs=90, batch_size=5)

import numpy as np
xx=[[6.7,3.3,5.7,2.5]]
px=model.predict(xx) #모델에 적용
pre=np.argmax(px) #argmax:집합 X 안에서 최대값의 위치
print(pre)
Y_col=np.unique(y) #열 이름 중복 제거
print(Y_col)
print(Y_col[pre])