import os
import numpy as np
import tensorflow as tf

from utils import FLAGS


def get_files(file_dir):
    D= []

    for (dirpath, dirnames, filenames) in os.walk(file_dir):
        for filename in filenames:
            D += [os.path.join(dirpath, filename)]

    temp = np.array([D])
    temp = temp.transpose()

    lists = list(temp[:,0])

    image_list = []

    for i in range(len(lists) - FLAGS.time_step):
        image_list.extend(lists[i:i+FLAGS.time_step])

    # print(image_list)
    return image_list


def get_batch(image, image_H, image_W, batch_size,capacity):
    image = tf.cast(image, tf.string)

    #加入队列
    input_queue = tf.train.slice_input_producer([image])

    #jpeg或者jpg格式都用decode_jpeg函数，其他格式可以去查看官方文档
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels=1)

    image = tf.image.resize_image_with_crop_or_pad(image, image_H, image_W)

    image = tf.image.per_image_standardization(image)
    #对resize后的图片进行标准化处理

    image_batch = tf.train.batch([image],batch_size = batch_size,num_threads=16,capacity = capacity)

    images_batch = tf.cast(image_batch, tf.float32)

    # y = get_label(0)

    return images_batch

def get_batch2(image_H, image_W, batch_size,capacity):
    # **1.把所有的 tfrecord 文件名列表写入队列中
    filename_queue = tf.train.string_input_producer(['drive/SRCN/data/800r_png_training.tfrecord'], num_epochs=1,
                                                    shuffle=False)
    # **2.创建一个读取器
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # **3.根据你写入的格式对应说明读取的格式
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string)
                                       }
                                       )
    img = features['image']
    # 这里需要对图片进行解码
    img = tf.image.decode_png(img, channels=1)  # 这里，也可以解码为 1 通道
    img = tf.reshape(img, [ image_H,image_W, 1])  # 28*28*3
    img = tf.cast(img, tf.float32)
    print('img is', img)

    X_batch = tf.train.batch([img], batch_size=batch_size, capacity=capacity, num_threads=16)

    X_batch = tf.cast(X_batch, tf.float32)

    return X_batch

def get_label(start, label):


    label_list = np.zeros(shape=[FLAGS.batch_size, FLAGS.time_step, FLAGS.road_num])

    for i in range(FLAGS.batch_size):
        label_list[i] = label[start+1+i : start+1+i+FLAGS.time_step]

    # label_list = tf.cast(label_list, tf.float32)

    return label_list

if __name__ == '__main__':
    image_list = get_files('E:/test/')
    # get_batch(image_list,998,828,16,64)
    # get_label()



