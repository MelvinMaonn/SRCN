import os
import numpy as np
import tensorflow as tf




def get_files(file_dir):

    pathname = ['E:/test/0.jpg','E:/test/1.jpg','E:/test/2.jpg','E:/test/3.jpg','E:/test/4.jpg']

    # for (dirpath, dirnames, filenames) in os.walk(file_dir):
    #     for filename in filenames:
    #         pathname += [os.path.join(dirpath, filename)]

    filename_queue = tf.train.string_input_producer(pathname, shuffle=False, num_epochs=1)

    return filename_queue

    #
    # threads = tf.train.start_queue_runners(sess=sess)



def get_batch(image,image_W,image_H, batch_size, capacity):
    #tf.cast()用来做类型转换
    # image = tf.cast(image,tf.string)

    image_reader = tf.WholeFileReader()
    key, image = image_reader.read(image)
    image = tf.image.decode_jpeg(image, channels=1)

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    image_batch = tf.train.batch([image],batch_size = batch_size,num_threads=16,capacity = capacity)

    return image_batch


if __name__ == '__main__':

    train_dir = 'E:/test'

    image = get_files(train_dir)

    image_batch = get_batch(image,998, 828, 16, 64)

    with tf.Session() as sess:

        # tf.train.string_input_producer定义了一个epoch变量，要对它进行初始化
        tf.local_variables_initializer().run()
        # 使用start_queue_runners之后，才会开始填充队列
        threads = tf.train.start_queue_runners(sess=sess)
        print(type(image))
        # print(type(image.eval))
        # print(image.eval.dtype)
        # print(image.eval.shape)
        print(type(image_batch))
        print(type(image_batch.eval()))
        print(image_batch.eval().dtype)
        print(image_batch.eval().shape)



