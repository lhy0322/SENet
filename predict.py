import tensorflow as tf
import sonnet as snt
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import model_DeepSEQ
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score, recall_score, precision_score, \
    accuracy_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = '0'


batch_size = 64
learning_rate = 0.0005
num_epochs = 5

def drawAUPR(y_pred, y_label, figure_file, method_name):
    '''
        y_pred is a list of length n.  (0,1)
        y_label is a list of same length. 0/1
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    '''
    y_label = np.array(y_label)
    y_pred = np.array(y_pred)
    plt.figure(figsize=(5, 5))
    lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_pred)
    plt.plot(lr_recall, lr_precision, lw = 1,
             label= method_name + ' (AUPR=%0.3f)' % average_precision_score(y_label, y_pred), color='red')

    y_pred = np.load("result/competition/DeepSE_human.npy")
    lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_pred)
    plt.plot(lr_recall, lr_precision, lw=1,
             label="DeepSE" + ' (AUPR=%0.3f)' % average_precision_score(y_label, y_pred), color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    fontsize = 12
    plt.xlabel('Recall', fontsize = fontsize)
    plt.ylabel('Precision', fontsize = fontsize)
    plt.title('Precision Recall Curve')
    plt.legend(fontsize=9)
    plt.savefig(figure_file)
    plt.show()
    return


def drawAUC(y_pred, y_label, figure_file, method_name):
    '''
        y_pred is a list of length n.  (0,1)
        y_label is a list of same length. 0/1
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    '''
    y_label = np.array(y_label)
    y_pred = np.array(y_pred)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(y_label, y_pred)
    roc_auc[0] = auc(fpr[0], tpr[0])

    y_pred = np.load("result/competition/DeepSE_human.npy")
    fpr[1], tpr[1], _ = roc_curve(y_label, y_pred)
    roc_auc[1] = auc(fpr[1], tpr[1])

    lw = 1
    plt.figure(figsize=(5, 5))
    plt.plot(fpr[0], tpr[0],
         lw=lw, label= method_name + ' (AUC=%0.3f)' % roc_auc[0], color='red')
    plt.plot(fpr[1], tpr[1],
             lw=lw, label="DeepSE" + ' (AUC=%0.3f)' % roc_auc[1], color='blue')
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    # plt.xticks(font="Times New Roman",size=18,weight="bold")
    # plt.yticks(font="Times New Roman",size=18,weight="bold")
    fontsize = 12
    plt.title("Receiver Operating Characteristic curve")
    plt.xlabel('False Positive Rate', fontsize = fontsize)
    plt.ylabel('True Positive Rate', fontsize = fontsize)
    #plt.title('Receiver Operating Characteristic Curve', fontsize = fontsize)
    plt.legend(loc="lower right", fontsize=9)
    plt.savefig(figure_file)
    plt.show()
    return


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


def get_metric(y_test, y_test_pred, p):
    test_pred_class = y_test_pred >= p
    test_acc = accuracy_score(y_test, test_pred_class)
    test_auc = roc_auc_score(y_test, y_test_pred)
    test_aupr = average_precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, test_pred_class, pos_label=1)
    test_precision = precision_score(y_test, test_pred_class, pos_label=1)

    result = [round(test_auc, 3), round(test_aupr, 3), round(test_acc, 3), round(test_recall, 3),
              round(test_precision, 3)]

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


def plot_xy(x_values, label, title):
    """绘图"""
    # fig, ax = plt.subplots(2, 2)
    df = pd.DataFrame(x_values, columns=['x', 'y'])
    df['label'] = label

    plt.scatter(df['x'], df['y'], c=df['label'], cmap='Spectral',s=20)
    # plt.colorbar()
    plt.show()


optimizer = snt.optimizers.Adam(learning_rate=learning_rate)

# model = model_DeepSEQ.Enformer(channels=32, num_transformer_layers=3, num_heads=8)
model = model_DeepSEQ.Enformer(channels=64, num_transformer_layers=2, num_heads=16)

checkpoint = tf.train.Checkpoint(module=model)
checkpoint.restore("model/merge/epoch-3")
# checkpoint.restore("model/ablation/No_Transformer/epoch-2")


y_ture, y_predict = evaluate_model(model, dataset=get_dataset("dataset_seq/predict/human/test").
                                   batch(batch_size*2).prefetch(batch_size*2))
# print(y_predict.shape)
# y_predict.resize((1034*2, 9600//2))

# draw
# from sklearn.manifold import TSNE
#
# tsne = TSNE(n_components=2)
# x_tsne = tsne.fit_transform(y_predict)
# plot_xy(x_tsne, y_ture, "t-sne")


result = get_metric(y_ture, y_predict, 0.5)
print(result)

drawAUC(y_predict, y_ture, "figure/AUC.png", "DeepSEQ")
# drawAUPR(y_predict, y_ture, "figure/AUPR.png", "DeepSEQ")



