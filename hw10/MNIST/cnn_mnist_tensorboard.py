import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

LOGDIR = "./M"
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv_layer(input, size_kernel, size_in, size_out, name="convolutions"):
  with tf.name_scope(name):
    w = tf.Variable(weight_variable([size_kernel, size_kernel, size_in, size_out]), name="W")
    b = tf.Variable(bias_variable([size_out]), name="B")
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    return act

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def fc_layer(input, size_in, size_out, name="fc"):
  with tf.name_scope(name):
    w = tf.Variable(weight_variable([size_in, size_out]), name="W")
    b = tf.Variable(bias_variable([size_out]), name="B")
    act = tf.matmul(input, w) + b
    return act


def main():
    x = tf.placeholder(tf.float32, [None, 28*28])
    y_ = tf.placeholder(tf.float32, [None, 10])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    conv1 = conv_layer(x_image,5, 1, 32)
    pool1 = max_pool_2x2(conv1)
    conv2 = conv_layer(pool1, 5, 32, 64)
    pool2 = max_pool_2x2(conv2)

    flatted = tf.reshape(pool2, [-1, 7*7*64])
    fc1 = fc_layer(flatted, 7*7*64, 1024)
    fc1 = tf.nn.relu(fc1)

    keep_prob = tf.placeholder(tf.float32)
    fc1_drop = tf.nn.dropout(fc1, keep_prob)

    fc2 = fc_layer(fc1_drop, 1024, 10)

    with tf.Session() as sess:
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=fc2))

        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y_, 1))

        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(LOGDIR + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(LOGDIR + '/test', sess.graph)

        sess.run(tf.global_variables_initializer())
    
        def feed_dict(train):
            if train:
                xs, ys = mnist.train.next_batch(100)
                k = 0.5
            else:
                xs, ys = mnist.test.next_batch(100)
                # xs, ys = mnist.test.images, mnist.test.labels
                k = 1.0
            return {x: xs, y_: ys, keep_prob: k}

        for i in range(2000):
            if i % 10 == 0:  # Record summaries and test-set accuracy
                test_accuracy = accuracy.eval(feed_dict=feed_dict(False))
                summary, train_accuracy = sess.run([merged, accuracy], feed_dict=feed_dict(False))
                test_writer.add_summary(summary, i)
                print('step %d, testing accuracy %g' % (i, test_accuracy))
            else:  # Record train set summaries, and train
                train_accuracy = accuracy.eval(feed_dict=feed_dict(True))
                summary, train_accuracy = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
                # dprint('step %d, training accuracy %g' % (i, train_accuracy))


        train_writer.close()
        test_writer.close()


if __name__ == '__main__':
    main()
