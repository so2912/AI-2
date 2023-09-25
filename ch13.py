import pandas as pd
# 데이터 입력
df = pd.read_csv('./data/sonar3.csv', header=None)

# 일반 암석(0)과 광석(1)이 몇 개 있는지 확인합니다.
print(df[60].value_counts())

# 음파 관련 속성을 X로, 광물의 종류를 y로 저장합니다.
X = df.iloc[:,0:60]
y = df.iloc[:,60]
#X = df.iloc[:140,0:60]
#y = df.iloc[:140,60]
#X_test = df.iloc[140:,0:60]
#y_test =  df.iloc[140:,60]

from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# 학습 셋과 테스트 셋을 구분합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

# 모델을 설정합니다.
model = Sequential()
model.add(Dense(24,  input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델을 컴파일합니다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델을 실행합니다.
his=model.fit(X, y, epochs=200, batch_size=10)
score=model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])

# 모델을 저장합니다.
model.save('./data/model/my_model.keras')
del model #메모리 내의 모델 삭제

# 모델을 새로 불러옵니다.
model = load_model('./data/model/my_model.keras')

# 불러온 모델을 테스트셋에 적용해 정확도를 구합니다.
score=model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])