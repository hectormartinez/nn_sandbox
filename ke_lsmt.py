# bidirectional model
# https://github.com/fchollet/keras/issues/1629
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM
from keras.layers.recurrent import Recurrent
from keras.preprocessing import sequence

import argparse

w2i_dict = dict()  # default value for unknown words
l2i_dict = dict()



def read_tab_sep(file_name):
    current_words = []
    current_tags = []

    for line in open(file_name).readlines():
        line = line.strip()

        if line:
            if len(line.split("\t")) != 2:
                if len(line.split("\t")) == 1:
                    raise IOError("Issue with input file - doesn't have a tag or token?")
                else:
                    print("erroneous line: {} (line number: {}) ".format(line, i), file=sys.stderr)
                    exit()
            else:
                word, tag = line.split('\t')
            current_words.append(word)
            current_tags.append(tag)

        else:
            if current_words:
                yield (current_words, current_tags)
            current_words = []
            current_tags = []
    if current_tags != []:
        yield (current_words, current_tags)



def string2index(word,dictindex,update=True):
    if word in dictindex:
        return dictindex[word]
    elif update:
        dictindex[word] = len(dictindex.keys())+1
        return dictindex[word]
    else:
        return 0


def main():
    parser = argparse.ArgumentParser(description="""toy LSTM""")
    parser.add_argument("--train",default="data/pos_ud_en_dev.conll")
    parser.add_argument("--test",default="data/pos_ud_en_test.conll")

    args = parser.parse_args()
    word_type_counter = 1

    train_X = []
    train_Y = []
    for wordseq, labelseq in read_tab_sep(args.train):
        train_x = [string2index(w,w2i_dict) for w in wordseq]
        train_y = [string2index(w,l2i_dict) for w in labelseq]
        train_X.append(train_X)
        train_Y.append(train_Y)

    test_X = []
    test_Y = []
    for wordseq, labelseq in read_tab_sep(args.test):
        test_x = [string2index(w, w2i_dict) for w in wordseq]
        test_y = [string2index(w, l2i_dict) for w in labelseq]
        test_X.append(test_x)
        test_Y.append(test_y)

    max_features = 20000
    max_length = 80
    embedding_dim = 256
    batch_size = 128
    epochs = 10

    model = Sequential()
    model.add(Embedding(max_features, embedding_dim, input_length=max_length, dropout=0.2))
    model.add(LSTM(embedding_dim, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(train_X, train_Y,batch_size=batch_size, nb_epoch=epochs, validation_data=(test_X, test_Y))


if __name__ == "__main__":
    main()