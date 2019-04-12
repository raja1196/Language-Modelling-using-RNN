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

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

#Given a strong will split string at the word level and return a cleaned list of word tokens
def clean_doc(doc):
	# replace '--' with a space ' '
	doc = doc.replace('--', ' ')
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# make lower case
	tokens = [word.lower() for word in tokens]
	return tokens

# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

def format_data():
    # load ascii text and covert to lowercase
    filename = r"simple-examples/data/ptb.train.txt"
    raw_text = load_doc(filename)
    raw_text = raw_text.lower()
    tokens = clean_doc(raw_text)
    print(tokens[:10])
    print('Total Tokens: %d' % len(tokens))
    print('Unique Tokens: %d' % len(set(tokens)))

    #Create list of tokens?
    length = 50 + 1
    sequences = list()
    #Value used to decrease number of training sequences
    div = 5
    for i in range(length, int(len(tokens)/div)):
        # select sequence of tokens
        seq = tokens[i-length:i]
        # convert into a line
        line = ' '.join(seq)
        # store
        sequences.append(line)
    print('Total Sequences: %d' % len(sequences))

    # save sequences to file
    out_filename = 'word_sequences.txt'
    save_doc(sequences, out_filename)
    return

def main():
    # load
    in_filename = 'word_sequences.txt'
    doc = load_doc(in_filename)
    lines = doc.split('\n')

    # integer encode sequences of words
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)

    # vocabulary size
    vocab_size = len(tokenizer.word_index) + 1

    # separate into input and output
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
    epoch =8
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
    model.save('word_level_model.h5')
    # save the tokenizer
    dump(tokenizer, open('tokenizer.pkl', 'wb'))
    return 

def generate_text():
    # load cleaned text sequences
    in_filename = 'word_sequences.txt'
    doc = load_doc(in_filename)
    lines = doc.split('\n')
    seq_length = len(lines[0].split()) - 1
    
    # load the model

    model = load_model('word_level_model.h5',custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform})
    print("hello")
    # load the tokenizer
    tokenizer = load(open('tokenizer.pkl', 'rb'))
    
    # select a seed text
    seed_text = lines[random.randint(0,len(lines))]
    print(seed_text + '\n')
    
    # generate new text
    generated = generate_seq(model, tokenizer, seq_length, seed_text, 50)
    print(generated)
    return


# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
		result.append(out_word)
	return ' '.join(result)


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