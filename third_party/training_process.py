import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
#import configParser
import random
import matplotlib.pyplot as plt
from tensorflow.python.framework import graph_util

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


config = configparser.ConfigParser()
config.read("./config.cfg")

is_test = True

Num_Para = 67
Num_Result = 4
#Num_Para = config.getint('Section1','Num_Para')
#Num_Result = config.getint('Section1','Num_Result')

while 1:
    nacc = input('traindata:\n')
    if os.path.isfile('processed/' + nacc + '_data.npy') and os.path.isfile('processed/' + nacc + '_label.npy'):
        break
    else:
        print('File not found, please retype it\n')
while 1:
    ntest = input('testdata:\n\t(Enter \'s\' to skip)\n')
    if ntest == "s":
        is_test = False
        break
    if os.path.isfile('processed/' + ntest + '_data.npy') and os.path.isfile('processed/' + ntest + '_label.npy'):
        break
    else:
        print('File not found, please retype it\n')
train_data = np.load(r'processed/' + nacc + '_data.npy')
train_label = np.load(r'processed/' + nacc + '_label.npy')
if is_test:
    test_data = np.load(r'processed/' + ntest + '_data.npy')
    test_label = np.load(r'processed/' + ntest + '_label.npy')
train_col = train_label.shape[0]
print('data loaded\n')

dataset_size = len(train_label)
print('dataset size:' + str(dataset_size) + '\n')



x = tf.placeholder('float', [None, 2*Num_Para], name='input')
y = tf.placeholder('float', [None, Num_Result], name='label')
weight = {
    'w1': tf.Variable(tf.random_normal([2*Num_Para, 40], stddev=0.3, dtype=tf.float32)),
    'w2': tf.Variable(tf.random_normal([40, 10], stddev=0.1, dtype=tf.float32)),
    'out': tf.Variable(tf.random_normal([10, Num_Result], stddev=0.1, dtype=tf.float32))}
biases = {
    'b1': tf.Variable(tf.random_normal([40])),
    'b2': tf.Variable(tf.random_normal([10])),
    'out': tf.Variable(tf.random_normal([Num_Result]))}


def multilayer_perceptron(X, Weight, Biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, Weight['w1']), Biases['b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, Weight['w2']), Biases['b2']))
    return tf.add(tf.matmul(layer_2, Weight['out']), Biases['out'],name='output')


actv = multilayer_perceptron(x, weight, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=actv, name='cost'))
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
prep = tf.equal(tf.arg_max(actv, 1), tf.arg_max(y, 1))
accr = tf.reduce_mean(tf.cast(prep, 'float'))
x_p = []
y_p = []
z_p = []
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    tf.train.write_graph(sess.graph_def, "models/"+nacc, "nn_model.pbtxt", as_text=True)
    checkpoint_path = os.path.join("models/"+nacc, "nn_model.ckpt")
    training_epoches = 200
    batch_size = 8
    diaplay_step = 1
    for epoch in range(training_epoches):
        batch_num = int(dataset_size / batch_size)
        avg_cost = 0
        for i in range(batch_num):
            start = (i * batch_size) % dataset_size
            end = min(start + batch_size, dataset_size)

            feeds = {x: train_data[start:end], y: train_label[start:end]}
            # print(batch_xs.shape,batch_ys.shape)
            sess.run(optm, feed_dict=feeds)
            avg_cost += sess.run(cost, feed_dict=feeds)
        # print(avg_cost,batch_num)
        avg_cost = avg_cost / batch_num
        x_p.append(epoch)
        if (epoch + 1) % diaplay_step == 0:
            print('epoch %03d/%03d cost:%.9f' % (epoch, training_epoches, avg_cost))
            train_accr = sess.run(accr, feed_dict=feeds)
            print('train accr:%.3f' % (train_accr))
            if is_test:
                print('test accr:%.3f' % (sess.run(accr, feed_dict={x: test_data, y: test_label})))
            y_p.append(avg_cost)
            z_p.append(train_accr)
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.plot(x_p,y_p)
    plt.show()
    #saver.save(sess, "model/model.ckpt")
    saver.save(sess, checkpoint_path)
    for tensor_name in tf.contrib.graph_editor.get_tensors(tf.get_default_graph()):
        print(tensor_name)
    print('done')