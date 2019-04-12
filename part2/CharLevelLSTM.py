# -*- coding: utf-8 -*-
import numpy
import string
import argparse
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import random 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers


from sklearn.model_selection import GridSearchCV

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import CustomObjectScope

from pickle import dump,load

def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

def format_data():
    # load text
    raw_text = load_doc("simple-examples/data/ptb.char.train.txt")
    # clean
    tokens = raw_text.split()
    print('Total Tokens: %d' % len(tokens))
    print('Unique Tokens: %d' % len(set(tokens)))

    raw_text = ''.join(tokens)

    # organize into sequences of characters
    length = 50
    sequences = list()
    div = 100
    for i in range(length, int(len(raw_text)/div)):
        # select sequence of tokens
        seq = raw_text[i-length:i+1]
        # store
        sequences.append(seq)

    # save sequences to file
    out_filename = 'char_sequences.txt'
    save_doc(sequences, out_filename)
    return

def main():
    # load
    in_filename = 'char_sequences.txt'
    raw_text = load_doc(in_filename)
    lines = raw_text.split('\n')
    
    #Encode sequences
    chars = sorted(list(set(raw_text)))
    mapping = dict((c, i) for i, c in enumerate(chars))

    sequences = list()
    for line in lines:
	# integer encode line
        encoded_seq = [mapping[char] for char in line]
        # store
        sequences.append(encoded_seq)

    # vocabulary size
    vocab_size = len(mapping)

    print('Vocabulary Size: %d' % vocab_size)
    
    sequences = numpy.asarray(sequences)
    X, y = sequences[:,:-1], sequences[:,-1]

    y = to_categorical(y, num_classes=vocab_size)
    seq_length = X.shape[1]

    ADAM = optimizers.Adam(lr = 0.001)
    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    
    #Hyper-parameters
    epoch = 5
    batch = 128
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=ADAM, metrics=['accuracy'])
    # fit model
    history = model.fit(X, y, batch_size=batch,validation_split = 0.25, epochs=epoch)

    train_perplexity=numpy.exp(history.history['loss'])
    val_perplexity=numpy.exp(history.history['val_loss'])
    plt.figure(1)
    plt.plot(range(epoch),train_perplexity)
    plt.plot(range(epoch),val_perplexity)
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.title('Training and Validation Perplexity')
    plt.legend(labels=['Training','Validation'],loc='best')
    plt.show()

    # save the model to file
    model.save('char_level_model.h5')
    # save the tokenizer
    dump(mapping, open('char_tokenizer.pkl', 'wb'))

    return

def generate_text():
    # load cleaned text sequences
    in_filename = 'char_sequences.txt'
    doc = load_doc(in_filename)
    lines = doc.split('\n')
    seq_length = len(lines[0].split()) - 1
    # load the model
    
    model = load_model('char_level_model.h5',custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform})
    # load the tokenizer
    mapping = load(open('char_tokenizer.pkl', 'rb'))
    seed = lines[numpy.random.randint(1, 50)]
    print(seed)
    print(generate_seq(model, mapping, seq_length, seed, 1000))  
    
    return

# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
	# generate a fixed number of characters
    for _ in range(n_chars):
        
		# encode the characters as integers
        encoded = [mapping[char] for char in in_text]
        encoded = encoded[-50:]
		# truncate sequences to a fixed length
        encoded = numpy.asarray(encoded)
		#encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# one hot encode
        encoded = encoded.reshape(1, encoded.shape[0])
		# predict character
        yhat = model.predict_classes(encoded, verbose=0)
		# reverse map integer to character
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
        numpy.append(encoded,char)
        
		# append to input
        in_text += char
        
    return in_text

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', action='store_true', help='Format text training data')
    parser.add_argument('--train', action='store_true', help='Used to train LSTM model')
    parser.add_argument('--generate', action='store_true', help='Used to generate text from model')
    return parser.parse_args()

if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.format:
        format_data()
    if FLAGS.train:
        main()
    if FLAGS.generate:
        generate_text()