# Google Colaboratory에서 Google Drive에 접근하는데 사용함
from google.colab import drive

# Google Drive를 /content/gdrive 디렉토리에 접근함
drive.mount('/content/gdrive')

#  현재 작업 디렉토리를 /content/gdrive/My Drive/data/fruit_dataset/ 디렉토리로 변경함
%cd /content/gdrive/My Drive/data/fruit_dataset/

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ImageDataGenerator를 생성함. (이미지 픽셀값을 0~1사이 범위로 조정)
train_datagen = ImageDataGenerator(rescale = 1./255)

# 주어진 폴더(train)에서 이미지 데이터를 불러와 전처리를 적용함.
training_set  = train_datagen.flow_from_directory('train',      #이미지 훈련 데이터 폴더명
                                   target_size = (64, 64),   # 이미지 크기 조정
                                   batch_size = 32,        # 한번에 32개 이미지 샘플수
                                   shuffle = True,        # 훈련 이미지 순서 섞음
                                   class_mode = 'categorical')  # 다중 클래스 모드

# ImageDataGenerator를 생성함. (이미지 픽셀값을 0~1사이 범위로 조정)
test_datagen = ImageDataGenerator(rescale = 1./255)

# 주어진 폴더(test)에서 이미지 데이터를 불러와 전처리를 적용함.
test_set = test_datagen.flow_from_directory('test',     # 이미지 테스트 데이터 폴더명
                                 target_size=(64, 64),  # 이미지 크기 조정
                                 shuffle = False,      # 테스트 이미지 순서 안바꿈
                                 class_mode='categorical')  # 다중 클래스 라벨

from keras.applications.vgg16 import VGG16

# vgg 모델 생성하기
vgg = VGG16(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
vgg.summary()

for layer in vgg.layers:
    layer.trainable = False

from keras.models import Sequential     # keras.model의 Sequential 불러오기
from keras.layers import Dense, Flatten, Input # keras.layers의 Dense 불러오기

model = Sequential()                       # 모델을 생성함
model.add(Input(shape=(64,64,3)))
model.add(vgg)                             # vgg 모델 추가함
model.add(Flatten())                       # 1차원 배열로 바꿈
model.add(Dense(64, activation='relu'))    # 은닉층 추가함
model.add(Dense(2, activation='softmax'))  # 출력층 추가함
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(training_set, epochs=10)

# 정확도(accuracy), 손실(loss) 그래프로 표현
import matplotlib.pyplot as plt

plt.xlabel('epoch')
plt.plot(history.history['accuracy'], label='training_accuracy')
plt.legend()
plt.show()


plt.xlabel('epoch')
plt.plot(history.history['loss'], label='training_loss')
plt.legend()
plt.show()

model.evaluate(test_set)

test_set.class_indices

predictions = model.predict(test_set)
predictions

import numpy as np
print(np.argmax(predictions, axis=1))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix  # 혼동행렬 함수를 불러옴

# 예측결과에서 가장 큰 값의 인덱스를 추출함
prediction = np.argmax(predictions, axis=1)
truth =  test_set.labels   # 테스트 데이터의 실제 레이블을 가져옴

# 혼동행렬을 계산함 (실제값, 예측값)
conf = confusion_matrix(truth, prediction)

# 혼동행렬을 표시함
plt.figure(figsize=(6, 4))  # 그림 크기 설정함
sns.heatmap(conf, annot=True, cmap="BuPu")

# 제목, x축 이름(Prediction), y축 이름(Truth)을 설정함
plt.title("Rotten Fruit Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


