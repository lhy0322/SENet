import tensorflow as tf
import sonnet as snt
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import model_DeepSEQ
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, accuracy_score
import itertools
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def get_tokenizer():
    f= ['a','c','g','t']
    c = itertools.product(f,f,f,f,f,f)
    res=[]
    for i in c:
        temp=i[0]+i[1]+i[2]+i[3]+i[4]+i[5]
        res.append(temp)
    res=np.array(res)
    NB_WORDS = 4097
    tokenizer = Tokenizer(num_words=NB_WORDS)
    tokenizer.fit_on_texts(res)
    acgt_index = tokenizer.word_index
    acgt_index['null']=0
    return tokenizer


def sentence2word(str_set):
    word_seq=[]
    for sr in str_set:
        tmp=[]
        for i in range(len(sr)-5):
            if('N' in sr[i:i+6]):
                tmp.append('null')
            else:
                tmp.append(sr[i:i+6])
        word_seq.append(' '.join(tmp))
    return word_seq


def word2num(wordseq,tokenizer,MAX_LEN):
    sequences = tokenizer.texts_to_sequences(wordseq)
    numseq = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    return numseq


def sentence2num(str_set,tokenizer,MAX_LEN):
    wordseq=sentence2word(str_set)
    numseq=word2num(wordseq,tokenizer,MAX_LEN)
    return numseq


def get_data(train,test):
    tokenizer=get_tokenizer()
    MAX_LEN=3000
    X_tr=sentence2num(train,tokenizer,MAX_LEN)
    X_te=sentence2num(test,tokenizer,MAX_LEN)

    return X_tr,X_te


def create_dataset(Kfold, type):

    print('Create dataset...')

    dataPath = './dataset_seq/crossvalid'
    train = './data/datasets/train_' + type + '.csv'
    train_csv = pd.read_csv(train)

    train_seq_num = 0
    test_seq_num = 0

    for i in range(Kfold):
        if not os.path.exists(dataPath + '/cross_' + str(i + 1)):
            os.makedirs(dataPath + '/cross_' + str(i + 1))

        train_writer = tf.io.TFRecordWriter(dataPath + '/cross_' + str(i + 1) + "/train")
        test_writer = tf.io.TFRecordWriter(dataPath + '/cross_' + str(i + 1) + "/valid")

        train_part = train_csv.drop(train_csv.index[(len(train_csv)//5) * i:(len(train_csv)//5) * (i+1)], inplace=False)
        test_part = train_csv[(len(train_csv)//5) * i:(len(train_csv)//5) * (i+1)]
        train_label = train_part['label'].tolist()
        train_seq = train_part['sequence'].tolist()
        test_label = test_part['label'].tolist()
        test_seq = test_part['sequence'].tolist()

        train_seq_num = len(train_part)
        test_seq_num = len(test_part)
        print(i, train_seq_num, test_seq_num)

        X_train, X_test = get_data(train_seq, test_seq)

        for i in range(train_seq_num):
            sequence = X_train[i]

            feature = {
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(train_label[i])])),
                'sequence': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=sequence))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            train_writer.write(example.SerializeToString())
        train_writer.close()

        for i in range(test_seq_num):
            sequence = X_test[i]

            feature = {
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(test_label[i])])),
                'sequence': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=sequence))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            test_writer.write(example.SerializeToString())
        test_writer.close()

    return train_seq_num, test_seq_num


def get_dataset(dataPath):

    dataset = tf.data.TFRecordDataset([dataPath], num_parallel_reads=8)
    dataset = dataset.map(_parse_function, 8)

    return dataset


def _parse_function(record):
    # 定义一个特征词典，和写TFRecords时的特征词典相对应
    features = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'sequence': tf.io.FixedLenFeature([3000], tf.int64),
    }
    example = tf.io.parse_single_example(record, features)

    sequence = tf.cast(example['sequence'], tf.int64)
    label = tf.cast(example['label'], tf.int64)

    return {
        'sequence': sequence,
        'label': label
    }


def create_step_function(model, optimizer):
    @tf.function
    def train_step(batch, optimizer_clip_norm_global=0.2):
        with tf.GradientTape() as tape:
            outputs = model(batch['sequence'], is_training=True)
            label = tf.expand_dims(batch['label'], axis=1)
            label = tf.cast(label, dtype=tf.float32)
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(label, outputs)
                # tf.nn.weighted_cross_entropy_with_logits(label, outputs, 3)
            )

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply(gradients, model.trainable_variables)

        return loss

    return train_step


def get_metric(y_test, y_test_pred):
    test_pred_class = y_test_pred >= 0.5
    test_acc = accuracy_score(y_test, test_pred_class)
    test_auc = roc_auc_score(y_test, y_test_pred)
    test_aupr = average_precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, test_pred_class, pos_label=1)
    test_precision = precision_score(y_test, test_pred_class, pos_label=1)

    result = [round(test_acc, 4), round(test_auc, 4), round(test_aupr, 4), round(test_recall, 4),
              round(test_precision, 4)]

    return result


def evaluate_model(model, dataset, max_steps=None):
    y_ture = np.array([])
    y_predict = np.array([])

    @tf.function
    def predict(x):
        return model(x, is_training=False)

    for i, batch in tqdm(enumerate(dataset)):
        if max_steps is not None and i > max_steps:
            break

        y_ture = np.append(y_ture, np.array(batch['label']))
        y_predict = np.append(y_predict, np.array(tf.nn.sigmoid(predict(batch['sequence']))))

    return y_ture, y_predict


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
batch_size = 64
learning_rate = 0.0005
num_epochs = 5
Kfold = 5
# type = 'human'
type = 'mouse'

train_seq_num, test_seq_num = create_dataset(Kfold, type)
# train_seq_num, test_seq_num = 16815, 4203

heads = [2, 4, 8, 16, 32]
channels = [16, 32, 64, 128, 256]
transformer_layer = [1, 2, 3, 4, 5, 6]

for param in heads:

    # result_writer = open("result/param/mouse/conv1_" + str(param) + ".txt", 'a')
    # result_writer = open("result/param/human/transformer2_" + str(param) + ".txt", 'a')
    # result_writer = open("result/param/human/head_" + str(param) + ".txt", 'a')
    result_writer = open("result/param/" + type + "/convolution_" + str(param) + ".txt", 'a')
    result_writer.write('ACC' + '\t' + 'AUC' + '\t' + 'AUPR' + '\t' + 'REC' + '\t' + 'PRE' + '\n')

    for fold in range(Kfold):
        bestResult = [0, 0, 0, 0, 0]

        train_dataset = get_dataset("dataset_seq/crossvalid/cross_" + str(fold + 1) + '/train'). \
            batch(batch_size).repeat().prefetch(batch_size * 2)

        model = model_DeepSEQ.Enformer(channels=64, num_transformer_layers=2, num_heads=8)
        optimizer = snt.optimizers.Adam(learning_rate=learning_rate)
        train_step = create_step_function(model, optimizer)

        steps_per_epoch = train_seq_num // batch_size

        # Train the model
        data_it = iter(train_dataset)
        for epoch_i in range(num_epochs):
            for i in tqdm(range(steps_per_epoch)):
                batch_train = next(data_it)
                loss = train_step(batch=batch_train)

            # ## Evaluate
            y_ture, y_predict = evaluate_model(model,
                                               dataset=get_dataset(
                                                   "dataset_seq/crossvalid/cross_" + str(fold + 1) + '/valid').
                                               batch(batch_size * 2).prefetch(batch_size * 2))
            result = get_metric(y_ture, y_predict)
            if result[1] > bestResult[1]:
                for i in range(5):
                    bestResult[i] = result[i]
            print("fold", fold, "epoch", epoch_i, ":", result)

        result_writer.write(
            str(bestResult[0]) + '\t' + str(bestResult[1]) + '\t' + str(bestResult[2]) + '\t' + str(
                bestResult[3]) + '\t'
            + str(bestResult[4]) + '\n')
        result_writer.flush()
    result_writer.close()




