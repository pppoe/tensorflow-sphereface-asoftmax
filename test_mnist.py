import tensorflow as tf
import numpy as np
import sys
from tensorflow.examples.tutorials.mnist import input_data
import tflearn
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from Loss_ASoftmax import Loss_ASoftmax

def visual_feature_space(features,labels, num_classes, name_dict = None):
    ''' https://raw.githubusercontent.com/ShownX/mxnet-center-loss/master/utils.py '''

    if name_dict is None:
        name_dict = dict([(d, '{}'.format(d)) for d in range(num_classes)])

    num = len(labels)

    # draw
    palette = np.array(sns.color_palette("hls", num_classes))
    
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(features[:,0], features[:,1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    ax.axis('off')
    ax.axis('tight')
    
    # We add the labels for each digit.
    txts = []
    for i in range(num_classes):
        # Position of each label.
        xtext, ytext = np.median(features[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, name_dict[i])
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.show()
    return f, ax, sc, txts

def Network(data_input, training = True):
    x = tflearn.conv_2d(data_input, 32, 3, strides = 1, activation='prelu', weights_init = 'xavier')
    x = tflearn.conv_2d(x, 32, 3, strides = 2, activation='prelu', weights_init = 'xavier')
    x = tflearn.conv_2d(x, 64, 3, strides = 1, activation='prelu', weights_init = 'xavier')
    x = tflearn.conv_2d(x, 64, 3, strides = 2, activation='prelu', weights_init = 'xavier')
    x = tflearn.conv_2d(x, 128, 3, strides = 1, activation='prelu', weights_init = 'xavier')
    x = tflearn.conv_2d(x, 128, 3, strides = 2, activation='prelu', weights_init = 'xavier')
    x = tflearn.flatten(x)
    feat = tflearn.fully_connected(x, 2, weights_init = 'xavier')
    return feat

class Module(object):

    def __init__(self, batch_size, num_classes):
        x = tf.placeholder(tf.float32, [batch_size, 784])
        y_ = tf.placeholder(tf.int64, [batch_size,])
        I = tf.reshape(x, [-1, 28, 28, 1])
        feat = Network(I)
        dim = feat.get_shape()[-1]
        logits, loss = Loss_ASoftmax(x = feat, y = y_, l = 1.0, num_cls = num_classes, m = 2)
        self.x_ = x
        self.y_ = y_
        self.y = tf.argmax(logits, 1)
        self.feat = feat
        self.loss = loss
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_), 'float'))

batch_size = 256
num_iters = 2000
num_classes = 10

mnist = input_data.read_data_sets('/tmp/MNIST', one_hot=False)
sess = tf.InteractiveSession()

mod = Module(batch_size, num_classes)
global_step = tf.Variable(0, trainable = False)
learning_rate = tf.train.exponential_decay(0.001, global_step, 10000, 0.9, staircase=True)
opt = tf.train.AdamOptimizer(learning_rate)
train_op = opt.minimize(mod.loss, global_step)

tf.global_variables_initializer().run()

for t in range(num_iters):

    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    fd = { mod.x_ : batch_xs, mod.y_ : batch_ys }
    _, v = sess.run([train_op, mod.loss], feed_dict=fd)
    if t % 100 == 0:
        print (t, v)

print ('Training Done')

### evaluation
num = mnist.test.images.shape[0]
test_data = np.ndarray([batch_size,784])
test_labels = np.ndarray([batch_size])

acc_vec = []
for b in range(0, num, batch_size):
    e = min([b+batch_size, num])
    sz = e - b
    test_data[0:sz,:] = mnist.test.images[b:e]
    test_labels[0:sz] = mnist.test.labels[b:e]
    acc_vec.append(sess.run(mod.accuracy, feed_dict={mod.x_: test_data, mod.y_: test_labels}))

print ('Testing Accuracy: ', np.mean(np.array(acc_vec)))

### visualize results
sample_data = np.array(range(mnist.test.images.shape[0]))
np.random.shuffle(sample_data)
sample_data = sample_data[:batch_size]
test_data = mnist.test.images[sample_data,:]
test_labels = mnist.test.labels[sample_data]
feat_vec = sess.run(mod.feat, feed_dict={mod.x_: test_data})
visual_feature_space(feat_vec, test_labels, num_classes)
