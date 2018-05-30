import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from model.srcn import SRCN
import data.read_data as rdd
from utils import FLAGS
from sklearn.metrics import mean_absolute_error

# def train(train_dir=None, val_dir=None, mode='train'):
#     model = SRCN(mode)
#     model.buildmodel()
#
#     config = tf.ConfigProto(allow_soft_placement=True)
#     config.gpu_options.allow_growth = True
#     with tf.Session(config=config) as sess:
#         sess.run(tf.global_variables_initializer())
#
#         saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
#         train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
#         if FLAGS.restore:
#             ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
#             if ckpt:
#                 # the global_step will restore sa well
#                 saver.restore(sess, ckpt)
#                 print('restore from checkpoint{0}'.format(ckpt))
#
#         print('=============================begin training=============================')
#         for cur_epoch in range(FLAGS.num_epochs):
#             train_cost = 0
#             start_time = time.time()
#             batch_time = time.time()
#
#             # the training part
#             for cur_batch in range(6123):
#                 if (cur_batch + 1) % 100 == 0:
#                     print('batch', cur_batch, ': time', time.time() - batch_time)
#                 batch_time = time.time()
#                 batch_inputs, _, batch_labels = \
#                     train_feeder.input_index_generate_batch(indexs)
#                 # batch_inputs,batch_seq_len,batch_labels=utils.gen_batch(FLAGS.batch_size)
#                 feed = {model.inputs: batch_inputs,
#                         model.labels: batch_labels}
#
#                 # if summary is needed
#                 summary_str, batch_cost, step, _ = \
#                     sess.run([model.merged_summay, model.cost, model.global_step, model.train_op], feed)
#                 # calculate the cost
#                 train_cost += batch_cost * FLAGS.batch_size
#
#                 train_writer.add_summary(summary_str, step)
#
#                 # save the checkpoint
#                 if step % FLAGS.save_steps == 1:
#                     if not os.path.isdir(FLAGS.checkpoint_dir):
#                         os.mkdir(FLAGS.checkpoint_dir)
#                     logger.info('save checkpoint at step {0}', format(step))
#                     saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'), global_step=step)
#
#                 # train_err += the_err * FLAGS.batch_size
#                 # do validation
#                 if step % FLAGS.validation_steps == 0:
#                     acc_batch_total = 0
#                     lastbatch_err = 0
#                     lr = 0
#                     for j in range(num_batches_per_epoch_val):
#                         indexs_val = [shuffle_idx_val[i % num_val_samples] for i in
#                                       range(j * FLAGS.batch_size, (j + 1) * FLAGS.batch_size)]
#                         val_inputs, _, val_labels = \
#                             val_feeder.input_index_generate_batch(indexs_val)
#                         val_feed = {model.inputs: val_inputs,
#                                     model.labels: val_labels}
#
#                         dense_decoded, lastbatch_err, lr = \
#                             sess.run([model.dense_decoded, model.cost, model.lrn_rate],
#                                      val_feed)
#
#                         # print the decode result
#                         ori_labels = val_feeder.the_label(indexs_val)
#                         acc = utils.accuracy_calculation(ori_labels, dense_decoded,
#                                                          ignore_value=-1, isPrint=True)
#                         acc_batch_total += acc
#
#                     accuracy = (acc_batch_total * FLAGS.batch_size) / num_val_samples
#
#                     avg_train_cost = train_cost / ((cur_batch + 1) * FLAGS.batch_size)
#
#                     # train_err /= num_train_samples
#                     now = datetime.datetime.now()
#                     log = "{}/{} {}:{}:{} Epoch {}/{}, " \
#                           "accuracy = {:.3f},avg_train_cost = {:.3f}, " \
#                           "lastbatch_err = {:.3f}, time = {:.3f},lr={:.8f}"
#                     print(log.format(now.month, now.day, now.hour, now.minute, now.second,
#                                      cur_epoch + 1, FLAGS.num_epochs, accuracy, avg_train_cost,
#                                      lastbatch_err, time.time() - start_time, lr))

# def main(_):
#     if FLAGS.num_gpus == 0:
#         dev = '/cpu:0'
#     elif FLAGS.num_gpus == 1:
#         dev = '/gpu:0'
#     else:
#         raise ValueError('Only support 0 or 1 gpu.')
#
#     with tf.device(dev):
#         if FLAGS.mode == 'train':
#             train(FLAGS.train_dir, FLAGS.val_dir, FLAGS.mode)
#
#         elif FLAGS.mode == 'infer':
#             infer(FLAGS.infer_dir, FLAGS.mode)


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

    x = rdd.get_batch2(FLAGS.image_height, FLAGS.image_width, FLAGS.batch_size * FLAGS.time_step, 100)

    label = np.genfromtxt('drive/SRCN/November_800r_velocity_cnn.txt')
    # label = np.genfromtxt('E:/data/Data0/output4/SRCN/November_800r_velocity_cnn.txt')

    with tf.Session() as sess:

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("drive/SRCN/logs", sess.graph)
        # writer = tf.summary.FileWriter("logs", sess.graph)


        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:

            for i in range(((int)((30630 - 12 - 3) / 5))*20):

                if coord.should_stop():
                    break

                y = rdd.get_label(i, label)

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

                _, loss, state, pred = sess.run([train_op, srcn.losses, srcn.lstm2.cell_final_state, srcn.pred], feed_dict=feed_dict)

                print('i is: '+str(i))
                # print('cost: ', round(cost, 4))
                print('loss: ', loss)
                # mae = mean_absolute_error(y_true=y.reshape(-1), y_pred= pred.reshape(-1))
                # print("mae: ", mae)
                print(sess.run(lrn_rate))
                # tf.summary.scalar('mae',mae)
                tf.summary.scalar('lrn_rate',lrn_rate)

                if i % 100 == 0:
                    rs = sess.run(merged, feed_dict=feed_dict)
                    writer.add_summary(rs, i)

        except tf.errors.OutOfRangeError:
            print('Done training epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads=threads)
