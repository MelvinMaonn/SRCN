import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model.srcn import SRCN
import data.read_data2 as rdd
from utils import FLAGS



if __name__ == '__main__':


    srcn = SRCN('train')
    srcn.buildmodel()
    srcn.compute_cost()
    train_op = tf.train.RMSPropOptimizer(learning_rate=FLAGS.initial_learning_rate).minimize(srcn.cost)

    image = rdd.get_files('E:/test/')
    x = rdd.get_batch(image, FLAGS.image_width, FLAGS.image_height, FLAGS.batch_size * FLAGS.time_step, 100)

    with tf.Session() as sess:

        # writer = tf.summary.FileWriter("logs", sess.graph)

        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:

            for i in range(10):

                if coord.should_stop():
                    break


                # sess.run(x)
                if i == 0:
                    x = x.eval()
                y = rdd.get_label(i)
                # y = tf.reshape(y, [-1, FLAGS.time_step, FLAGS.road_num])
                # y = y.eval()
                #
                # print("convert")
                #
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
                        # srcn.cell_init_state: state  # use last state as the initial state for this run
                    }

                _, cost = sess.run([train_op,srcn.cost], feed_dict=feed_dict)

                print('cost: ', round(cost, 4))

        except tf.errors.OutOfRangeError:
            print('Done training epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads=threads)