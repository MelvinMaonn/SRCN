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

    image = rdd.get_files('E:/realDataFormat_800r')
    x = rdd.get_batch(image, FLAGS.image_height, FLAGS.image_width, FLAGS.batch_size * FLAGS.time_step, 100)


    with tf.Session() as sess:

        # writer = tf.summary.FileWriter("logs", sess.graph)

        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        x = x.eval()

        try:

            for i in range(510):

                if coord.should_stop():
                    break

                y = rdd.get_label(i*FLAGS.batch_size*FLAGS.time_step)

                if i == 0:
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

                _, cost, state, pred = sess.run([train_op, srcn.cost, srcn.lstm2.cell_final_state, srcn.pred], feed_dict=feed_dict)

                print('cost: ', round(cost, 4))
                mae = mean_absolute_error(y_true=y.reshape(-1), y_pred= pred.reshape(-1))
                print("mae: ", mae)

        except tf.errors.OutOfRangeError:
            print('Done training epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads=threads)