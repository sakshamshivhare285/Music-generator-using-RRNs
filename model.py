import os
import tensorflow
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, TimeDistributed,LSTM, Dropout, Embedding,Activation

model_dir = './models'

def save_weights(epoch, model):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save_weights(os.path.join(model_dir,'weights.{}.h5'.format(epoch)))

def load_weights(epoch,model):
    model.load_weights(os.path.join(model_dir,'weights.{}.h5'.format(epoch)))

def build_model(batch_size, seq_len,vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size,512,batch_input_shape=(batch_size,seq_len)))
    for i in range(3):
        model.add(LSTM(256,return_sequences=True, stateful=True))
        model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))

    return model

if __name__ == "__main__":
    model = build_model(16,64,50)
    model.summary()