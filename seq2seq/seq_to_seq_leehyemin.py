# 투빅스 nlp 세미나 최종 과제 (seq_to_seq 구현)
# 투빅스 13기 이혜민

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

batch_size = 64
epochs = 60
latent_dim = 256
num_samples = 10000

input_texts, target_texts = [], []
input_characters = []
target_characters = []


# 한국어 데이터, 영어 데이터 불러오기
with open('korean-english-park.dev.ko','r', encoding='utf-8') as f :
  input_texts = f.read().split(' ')

with open('korean-english-park.dev.en','r', encoding='utf-8') as f :
  target_texts = f.read().split(' ')
  
  
# 한국어/영어 list split

for i in input_texts:
  for j in list(i):
    input_characters.append(j)

input_characters = [word.strip('\n') for word in input_characters]
input_characters = [word.strip('') for word in input_characters]

for i in target_texts:
  for j in list(i):
    target_characters.append(j)
target_characters = [word.strip('\n') for word in target_characters ]
target_characters = [word.strip('') for word in target_characters ]



# 랜덤으로 확인하기
print(random.sample(input_characters,10))
print(random.sample(target_characters,10))

# 최대 길이 값을 알아내고 들어간 단어의 개수를 파악한다.
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])


input_token_index =dict([(char,i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])


encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens),dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens),dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens),dtype='float32')


test_input_data=np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens),dtype='float32')


for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
  for t, char in enumerate(input_text):
    encoder_input_data[i, t, input_token_index[char]] = 1.
  for t, char in enumerate(target_texts):
    decoder_input_data[i, t, target_token_index[char]] = 1.

    if t>0:
      decoder_target_data[i, t-1, target_token_index[char]] = 1.
      
      
# 모델 생성
encoder_inputs = input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# 출력상태 벡터 가져오기
encoder_states = [state_h, state_c]


decoder_inputs = input(shape=(None, num_decoder_tokens))

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)



# 모델 summary, compile
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer = 'adam')
early_stop = EarlyStopping(monitor='val_loss', patience=3)
