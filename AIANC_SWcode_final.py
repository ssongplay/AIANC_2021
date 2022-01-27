###############################################
############ 2021 한이음 ICT멘토링 ############
########## AI를 이용한 생활소음 감소 ##########
###############################################

##### Beep음을 예측하는 코드 #####
# 실행 환경 : Jupyter Notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.models import load_model
import datetime
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
from pydub import AudioSegment
from scipy.signal import resample
import IPython.display as ipd

uploaded_file_name = 'Beep.wav'  # 업로드 파일명
EXPECTED_SAMPLE_RATE = 44100 #44.1KHz

# convert_audio_for_model : wav 파일을 모델 학습용 wav로 변환하는 함수
# 44.1KHz, 모노 채널
def convert_audio_for_model(user_file, output_file = 'converted_audio_file.wav'):
    audio = AudioSegment.from_file(user_file, format = "wav")
    audio = audio.set_frame_rate(EXPECTED_SAMPLE_RATE).set_channels(1)
    audio.export(output_file, format="wav")
    return output_file

converted_audio_file = convert_audio_for_model(uploaded_file_name)

# wav 파일을 재생해볼 수 있는 코드
ipd.Audio(converted_audio_file)

# Loading audio samples from the wav file :
sample_rate, audio_samples = wavfile.read(converted_audio_file, 'rb')

# 오디오 정보 출력
duration = len(audio_samples)/sample_rate
print("data :", audio_samples)
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(audio_samples)}')

# csv파일 생성하여 불러오기
pd.DataFrame(audio_samples).to_csv("Beep.csv")
data = pd.read_csv('Beep.csv')

# data 정보 출력
print(data['0'].describe())

# train data 그래프 출력
plt.plot(data['0'])
#plt.xlim(0,1000)
plt.legend()
plt.show()
data  # data 확인

# 모델 학습을 위한 X, y 생성
X = np.arange(len(data)).reshape(-1,1)
y = data['0'].values
y = y.reshape(-1,1)
print("X : \n", X)
print("\ny : \n", y)

# 딥러닝 학습을 정상적으로 동작 시키기 위해 데이터 정규화
y_std = y.std()
y_mean = y.mean()
y = (y-y_mean) / y_std
y

seq_len = 50   #window size : 예측을 위한 데이터 수
prediction = 1  # 다음을 예측할 데이터 수
sequence_length = seq_len + prediction

result = []
for index in range(len(y) - sequence_length):
    result.append(y[index: index + sequence_length])

# 현재를 기준으로 최근 50개 데이터를 시험에 사용하고 그 외의 이전 데이터들은 학습에 사용
test_period = 50

X_train = train[:, :-prediction]
y_train = train[:,-prediction]
X_test = result[row:, :-prediction]
y_test = result[row:, -prediction]
print("shape of X_train : ", X_train.shape)
print("shape of y_train : ", y_train.shape)
print("shape of X_test : ", X_test.shape)
print("shape of y_test : ", y_test.shape)

# 모델 학습을 위한 reshape
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
print("shape of X_train : ", X_train.shape)
print("shape of X_test : ", X_test.shape)

# 모델 생성
print('Build LSTM RNN model ...')
model = Sequential()

# LSTM layer
model.add(LSTM(50, return_sequences=True, input_shape=(50,1)))  # (timestep, feature)
model.add(LSTM(64, return_sequences=False))

# Output(Dense) : 1개
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop')
model.summary()

##### 모델 학습하는 부분 #####
# %%time 으로 학습시간 측정 가능
%%time
# 트레이닝 값으로 학습
print ("training started..... please wait.")
hist = model.fit(X_train, y_train, epochs = 5, batch_size=10, validation_data=(X_test, y_test))

# 모델 저장
model.save('weight.h5')
print ("training finised!")

# 학습 과정에서의 loss 측정한 그래프 출력
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show

# 모델 평가
trainScore = model.evaluate(X_train, y_train, verbose=0)
testScore = model.evaluate(X_test, y_test, verbose=0)
print('Train Score : ', trainScore)
print('Test Score : ', testScore)

pred = model.predict(X_test)  # 소음 예측한 값

# 실제 입력 값과 모델이 예측한 값 출력
fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, '.', label='Prediction')
ax.legend()
plt.xlim(0,350000)
plt.show()

# 정규화된 데이터를 복원하기 위해 사용하는 함수
def denorm(y):
    return y * y_std + y_mean

fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(denorm(y_test), label='True (denorm)')
ax.plot(denorm(pred), '.', label='Prediction (denorm)')
ax.legend()
plt.xlim(0,350000)
plt.show()

# 정규화된 데이터로 학습한 예측값을 복원
denorm_pred = denorm(pred)

# 예측한 pred값 wav파일로 저장
wavfile.write("pred.wav", EXPECTED_SAMPLE_RATE, pred)

ipd.Audio("pred.wav") # 예측값을 재생해볼 수 있는 코드

# 예측한 pred.wav를 csv로 변환
pd.DataFrame(denorm_pred).to_csv("Pred.csv")


###################################


##### 예측한 wav의 역위상을 출력하는 코드 #####
from pydub import AudioSegment
from pydub.playback import play
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import IPython.display as ipd

original_FileName = "converted_audio_file.wav"
pred_FileName = "pred.wav"
# 지정한 original wav 파일 load
original_sound = AudioSegment.from_file(original_FileName, format="wav")
# 지정한 pred wav 파일 load
pred_sound = AudioSegment.from_file(pred_FileName, format="wav")
# 예측 파일 역위상으로 뒤집기
pred_sound_reverse = pred_sound.invert_phase()
# 역위상 파장 wav파일로 저장
pred_sound_reverse.export("pred_sound_reverse.wav", format="wav")
# 두개의 소리(wav파일)를 결합(Merge)
combinedSound_AI = original_sound.overlay(pred_sound_reverse)
# 결합된 소리를 저장
combinedSound_AI.export("combined_AI.wav", format="wav")

# 원래 소음 재생
ipd.Audio(original_FileName)

# 원래 소음 정위상 + 예측한 소음의 역위상 재생
ipd.Audio("combined_AI.wav")


# 그래프 출력
fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)

ax2 = ax.twinx()
sample_rate1, audio_samples1 = wavfile.read("converted_audio_file.wav", 'rb')
pd.DataFrame(audio_samples1).to_csv("original.csv")
data1 = pd.read_csv("original.csv")
ax2.plot(data1['0'], color='red', label="original")

sample_rate2, audio_samples2 = wavfile.read("pred_sound_reverse.wav", 'rb')
pd.DataFrame(audio_samples2).to_csv("pred_sound_reverse.csv")
data2 = pd.read_csv("pred_sound_reverse.csv")
#ax.plot(data2['0'], color='green',label="pred_reverse")

sample_rate3, audio_samples3 = wavfile.read("combined_AI.wav", 'rb')
pd.DataFrame(audio_samples3).to_csv("combined_AI.csv")
data3 = pd.read_csv("combined_AI.csv")
ax.plot(data3['0'], color='blue',label="combined")

ax.legend(loc='upper right')
ax2.legend(loc='lower right')
#plt.xlim(253340,253440)
plt.xlim(252440,253440)
plt.show()
