import datetime

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from model.srcn import SRCN
import data.read_data as rdd
from utils import FLAGS
from model import metrics
from sklearn.metrics import mean_absolute_error

import os
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

logger = logging.getLogger('Traing for SRCN')
logger.setLevel(logging.INFO)

def train(mode='train'):
    srcn = SRCN(mode)
    srcn.buildmodel()
    srcn.compute_cost()
    global_step = tf.train.get_or_create_global_step()

    lrn_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                          global_step,
                                          FLAGS.decay_steps,
                                          FLAGS.decay_rate,
                                          staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=lrn_rate,
                                      beta1=FLAGS.beta1,
                                      beta2=FLAGS.beta2).minimize(srcn.losses,
                                                                  global_step=global_step)

    num_train_samples = 1021 * 21
    num_batches_per_epoch = int(num_train_samples / FLAGS.batch_size)
    num_val_samples = 1021 * 6
    num_batches_val = int(num_val_samples / FLAGS.batch_size)

    train_image, train_label = rdd.get_batch2('/home/mm/SRCN/data/data1/800r_train.tfrecord',FLAGS.image_height, FLAGS.image_width, FLAGS.batch_size * FLAGS.time_step, 100)
    # val_image, val_label = rdd.get_batch2('/home/mm/SRCN/data/data1/800r_validation.tfrecord',FLAGS.image_height, FLAGS.image_width, FLAGS.batch_size * FLAGS.time_step, 100)

    f = open('train_and_val_result.txt', 'w')

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("/home/mm/SRCN/logs", sess.graph)
        #train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                # the global_step will restore sa well
                saver.restore(sess, ckpt)
                print('restore from checkpoint{0}'.format(ckpt))

        print('=============================begin training=============================')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:

            for cur_epoch in range(FLAGS.num_epochs):
                start_time = time.time()

                # the training part
                for cur_batch in range(num_batches_per_epoch):

                    x, y = sess.run([train_image, train_label])

                    if cur_epoch == 0 and cur_batch == 0:

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


                    _, loss, state, pred = sess.run([train_op, srcn.losses, srcn.lstm2.cell_final_state, srcn.pred],
                                                    feed_dict=feed_dict)
                    # calculate the cost
                    #train_cost += batch_cost * FLAGS.batch_size

                    tf.summary.scalar('lrn_rate', lrn_rate)

                    if cur_batch % 1000 == 0:
                        rs = sess.run(merged, feed_dict=feed_dict)
                        writer.add_summary(rs, cur_batch)
                        print(str(cur_batch) + ":" + str(loss))

                # save the checkpoint
                if not os.path.isdir(FLAGS.checkpoint_dir):
                    os.mkdir(FLAGS.checkpoint_dir)
                    print('no checkpoint')
                logger.info('save checkpoint at step {0}', format(cur_epoch))
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'srcn-model.ckpt'), global_step=cur_epoch)

                '''
                for cur_batch in range(num_batches_val):

                    x, y = sess.run([val_image, val_label])

                    feed_dict = {
                        srcn.xs: x,
                        srcn.ys: y,
                    }

                    loss, pred = sess.run([srcn.losses, srcn.pred], feed_dict=feed_dict)

                    if cur_batch % 1000 == 0:
                        print(loss)
                '''

                now = datetime.datetime.now()
                log = "{}/{} {}:{}:{} Epoch {}/{}, " \
                      "loss = {:.3f}, " \
                      "time = {:.3f}"
                print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                                 cur_epoch + 1, FLAGS.num_epochs, loss, time.time() - start_time))

                f.write(log+'\r\n')

        except tf.errors.OutOfRangeError:
            print('Done training epoch limit reached')
        finally:
            coord.request_stop()
            f.close()

        coord.join(threads=threads)


def infer(mode='infer'):
    srcn = SRCN(mode)
    srcn.buildmodel()
    srcn.compute_cost()

    num_test_samples = 1021 * 3
    total_steps = int(num_test_samples / FLAGS.batch_size)

    test_image, test_label = rdd.get_batch2('/home/mm/SRCN/data/data1/800r_test.tfrecord',FLAGS.image_height, FLAGS.image_width, FLAGS.batch_size * FLAGS.time_step, 100)

    f = open('test_result.txt', 'w')

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print('restore from ckpt{}'.format(ckpt))
        else:
            print('cannot restore')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:

            start_time = time.time()

            for i in range(total_steps):

                if coord.should_stop():
                    break

                x, y = sess.run([test_image, test_label])

                feed_dict = {
                    srcn.xs: x,
                    srcn.ys: y,
                }

                loss, pred = sess.run([srcn.losses, srcn.pred], feed_dict=feed_dict)

                # for j in range(5):
                #     print(pred[j*12+11][144])

                if i % 100 == 0:
                    print(loss)

                # mae, mape, rmse = metrics.calculate_metrics(tf.reshape(srcn.pred, [-1]), tf.reshape(srcn.ys, [-1]))

                now = datetime.datetime.now()
                log = "{}/{} {}:{}:{} , " \
                      "loss = {:.3f}, " \
                      "time = {:.3f}"
                log = log.format(now.month, now.day, now.hour, now.minute, now.second,
                                                         loss, time.time() - start_time)
                print(log)
                f.write(log + '\r\n')

        except tf.errors.OutOfRangeError:
            print('Done training epoch limit reached')
        finally:
            coord.request_stop()
            f.close()

        coord.join(threads=threads)


def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(FLAGS.mode)

        elif FLAGS.mode == 'infer':
            infer(FLAGS.mode)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

'''
if __name__ == '__main__':


    srcn = SRCN('train')
    srcn.buildmodel()
    srcn.compute_cost()
    global_step = tf.train.get_or_create_global_step()
    lrn_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                               global_step,
                                               FLAGS.decay_steps,
                                               FLAGS.decay_rate,
                                               staircase=True)
    # train_op = tf.train.RMSPropOptimizer(learning_rate=lrn_rate, momentum=FLAGS.momentum).minimize(srcn.cost,global_step=global_step)
    train_op = tf.train.AdamOptimizer(learning_rate=lrn_rate,
                                            beta1=FLAGS.beta1,
                                            beta2=FLAGS.beta2).minimize(srcn.losses,
                                                                        global_step=global_step)

    x,y = rdd.get_batch2(FLAGS.image_height, FLAGS.image_width, FLAGS.batch_size * FLAGS.time_step, 100)

    # label = np.genfromtxt('drive/SRCN/November_800r_velocity_cnn.txt')
    # label = np.genfromtxt('E:/data/Data0/output4/SRCN/November_800r_velocity_cnn.txt')

    with tf.Session() as sess:

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("drive/SRCN/logs", sess.graph)
        # writer = tf.summary.FileWriter("logs", sess.graph)


        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:

            for i in range(2000):

                if coord.should_stop():
                    break

                # y = rdd.get_label(i, label)

                if i == 0:
                    x = x.eval()
                    y = y.eval()

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

                _, loss, state, pred = sess.run([train_op, srcn.losses, srcn.lstm2.cell_final_state, srcn.pred], feed_dict=feed_dict)

                # for j in range(5):
                #     print(pred[j*12+11][144])

                tf.summary.scalar('lrn_rate',lrn_rate)

                if i % 100 == 0:
                    rs = sess.run(merged, feed_dict=feed_dict)
                    writer.add_summary(rs, i)
                    print(loss)

        except tf.errors.OutOfRangeError:
            print('Done training epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads=threads)
'''
