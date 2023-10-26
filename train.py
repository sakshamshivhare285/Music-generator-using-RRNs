import numpy as np
import os
import json
import argparse

from model import build_model, save_weights

data_dir = './data'
log_dir = './logs'

batch_length = 16
seq_length = 64


class TrainLogger(object):
    def __init__(self, file):
        self.file = os.path.join(log_dir, file)
        self.epoch = 0
        with open(self.file, 'w') as f:
            f.write('epoch,loss,acc\n')

    def add_entry(self, loss, acc):
        self.epoch += 1
        s = '{},{},{}\n'.format(self.epoch, loss, acc)
        with open(self.file, 'a') as f:
            f.write(s)


def read_batches(T, vocab_size):
    length = T.shape[0]
    batch_char = int(length / batch_length)

    for start in range(0, batch_char - seq_length, seq_length):

        X = np.zeros((batch_length, seq_length))
        Y = np.zeros((batch_length, seq_length, vocab_size))
        for batch_idx in range(0, batch_length):

            for i in range(0, seq_length):
                X[batch_idx, i] = T[batch_char * batch_idx + start + i]
                Y[batch_idx, i, T[batch_char * batch_idx + start + i]] = 1
        yield X, Y

def train(text, epochs=50, save_freq=10):

    '''converting char to index and vice versa'''
    char_to_idx = {ch: i for (i, ch) in enumerate(sorted(list(set(text))))}
    print('number of unique characters are', str(len(char_to_idx)))

    with open(os.path.join(data_dir, 'char_to_idx.json'), 'w') as f:
        json.dump(char_to_idx, f)

    idx_to_char = {i: ch for (ch, i) in char_to_idx.items()}
    vocab_size = len(idx_to_char)

    '''Defining model architecture'''
    model = build_model(batch_length, seq_length, vocab_size)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    '''training data generation'''
    T = np.asarray([char_to_idx[c] for c in text], dtype=np.int32)
    print('Length of text is', T.size)

    steps_per_epoch = (len(text) / batch_length - 1) / seq_length

    log = TrainLogger('training_log.csv')

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        losses, accs = [], []

        for i, (X, Y) in enumerate(read_batches(T, vocab_size)):
            print(X)

            loss, acc = model.train_on_batch(X, Y)
            print('Batch {}:loss = {}, acc ={}'.format(i + 1, loss, acc))
            losses.append(loss)
            accs.append(acc)
        log.add_entry(np.average(losses), np.average(accs))

        if (epoch + 1) % save_freq == 0:
            save_weights(epoch + 1, model)
            print('Saved checkpoint to ', 'weights.{}.h5'.format(epoch + 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model on some text.')
    parser.add_argument('--input', default='input.txt', help='name of the text file to train from')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--freq', type=int, default=5, help='checkpoint save frequency')
    args = parser.parse_args()

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    train(open(os.path.join(data_dir, args.input)).read(),args.epochs,args.freq)
