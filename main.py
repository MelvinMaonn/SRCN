import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model.srcn import SRCN
import data.read_data2 as rdd
from utils import FLAGS
from sklearn.metrics import mean_absolute_error



if __name__ == '__main__':


    srcn = SRCN('train')
    srcn.buildmodel()
    srcn.compute_cost()
    train_op = tf.train.RMSPropOptimizer(learning_rate=FLAGS.initial_learning_rate).minimize(srcn.cost)

    x = rdd.get_batch2(FLAGS.image_height, FLAGS.image_width, FLAGS.batch_size * FLAGS.time_step, 100)

    label = np.genfromtxt('drive/SRCN/November_800r_velocity_cnn.txt')

    with tf.Session() as sess:

        # writer = tf.summary.FileWriter("logs", sess.graph)

        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:

            for i in range(6123):

                if coord.should_stop():
                    break

                y = rdd.get_label(i*FLAGS.batch_size, label)

                if i == 0:
                    x = x.eval()

                    feed_dict = {
                        srcn.xs: x,
                        srcn.ys: y,
                        # create initial state
                    }
                else:
                    feed_dict = {
                        srcn.xs: x,
                        srcn.ys: y,
                        # TODO 后期需要评估是否应该吧上一次的state给到下一次
                        srcn.lstm2.cell_init_state: state  # use last state as the initial state for this run
                    }

                _, cost,loss, state, pred = sess.run([train_op, srcn.cost,srcn.losses, srcn.lstm2.cell_final_state, srcn.pred], feed_dict=feed_dict)

                print('i is: '+str(i))
                print('cost: ', round(cost, 4))
                print('loss: ', loss)
                mae = mean_absolute_error(y_true=y.reshape(-1), y_pred= pred.reshape(-1))
                print("mae: ", mae)

        except tf.errors.OutOfRangeError:
            print('Done training epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads=threads)