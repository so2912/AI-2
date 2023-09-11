# -*- coding: utf-8 -*-
# 코드 내부에 한글을 사용가능 하게 해주는 부분입니다.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# 피마 인디언 당뇨병 데이터셋을 불러옵니다. 불러올 때 각 컬럼에 해당하는 이름을 지정합니다. 지정하지 않으면 첫줄이 이름이다.
df = pd.read_csv('pima-indians-diabetes.csv',
names = ["pregnant", "plasma", "pressure", "thickness",
         "insulin", "BMI", "pedigree", "age", "diabetes"])
# 임신횟수       포도당       혈압      피부주름두께   혈청-인슐린  체질량지수 가족력     나이     당뇨
pd.set_option('display.max_row', None)# 판다스 행을 무제한 확장
pd.set_option('display.max_columns', None) # 판다스 열을 무제한 확장
pd.set_option('display.width', 1000)#디스플레이 폭 확장
print(df)

pima=df.groupby(['pregnant'])
print(pima)
print(pima.count())

#정상과 당뇨 환자가 각각 몇 명씩인지 조사해 봅니다.
print(df["diabetes"].value_counts())

# 각 항목이 어느정도의 상관 관계를 가지고 있는지 알아봅니다.
print(df.corr())
'''
#colormap = plt.cm.gist_heat
colormap = plt.winter()
sns.heatmap(df.corr(),linewidths=0.1,vmax=0.5,cmap=colormap,annot=True,linecolor='red')
plt.show()

grid = sns.FacetGrid(df, col='diabetes')
grid.map(plt.hist, 'plasma', cumulative=True,  bins=10)
plt.show()
'''
plt.hist(x=[df.plasma[df.diabetes==0], df.plasma[df.diabetes==1]], alpha=0.5,histtype='stepfilled', bins=30,label=['normal','diabetes'])
      #histtype 옵션  'bar(default)','barstacked', 'step','stepfilled'
      # alpha (투명도) : 0.0 ~ 1.0사이
plt.legend()
plt.show()

dataset = df.values
X = dataset[0:500,0:8]
Y = dataset[0:500,8]
X_eval = dataset[500:,0:8]
Y_eval = dataset[500:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu', name='layer_1'))
model.add(Dense(8, activation='relu', name='layer_2'))
model.add(Dense(1, activation='sigmoid', name='layer_3'))
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#정확도, 1.XXX값은 1로 분류해버리는 분류모델 같은 경우에 사용하는 지표가 'accuracy'입니다.
#소수점을 사용하는 회귀 모델 같은 경우는 accuracy를 사용할 수 없습니다.

model.fit(X, Y, epochs=20, batch_size=10)

print (model.evaluate(X_eval, Y_eval,verbose=1)) #evaluate 평가