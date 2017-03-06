import tensorflow as tf
import numpy as np
from skimage import io
import read_data

train_root = '/data1/data/landmark/training/'
file_root = '/data1/data/landmark/training/filelist/sensetime1.txt'

data_images,data_points = read_data.read_data()


def addlayer(inputs,in_size,out_size,activation_function=None,layer_name=None):
    with tf.name_scope(layer_name):
        Weights = tf.Variable(tf.random_normal([in_size,out_size]))
        biases = tf.Variable(tf.zeros([1,out_size]))

    if(activation_function==None):
        outputs = tf.add(tf.matmul(inputs,Weights),biases)
    else:
        outputs = activation_function(tf.add(tf.matmul(inputs,Weights),biases))

    return(outputs)
with tf.device('/gpu:2'):
    image_inputs = tf.placeholder(dtype='float32',shape=[None,120,120,3],name='image_inputs')
    points = tf.placeholder(dtype='float32',shape=[None,50],name='points')
    with tf.name_scope('conv1'):
        l1 = tf.layers.conv2d(image_inputs,filters=16,kernel_size=[5,5],strides=(1,1),activation=tf.nn.relu,padding='VALID',name='conv_layer1')
        p1 = tf.nn.max_pool(l1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    with tf.name_scope('conv2'):
        l2 = tf.layers.conv2d(p1,filters=32,kernel_size=[5,5],strides=(1,1),activation=tf.nn.sigmoid,padding='VALID',name='conv_layer2')
        p2 = tf.nn.max_pool(l2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    with tf.name_scope('conv3'):
        l3 = tf.layers.conv2d(p2,filters=64,kernel_size=[5,5],strides=(2,2),activation=tf.nn.sigmoid,padding='VALID',name='conv_layer3')
        p3 = tf.nn.max_pool(l3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    with tf.name_scope('conv4'):
        l4 = tf.layers.conv2d(p3,filters=128,kernel_size=[3,3],strides=(1,1),activation=tf.nn.relu,padding='VALID',name='conv_layer4')

    l4_flat = tf.reshape(l4,shape=[-1,4*4*128])
    # p3_flat = tf.reshape(p3,shape=[-1,6*6*64])
    with tf.name_scope('fc1'):
        fc1_weights = tf.Variable(tf.random_normal([4*4*128,50]),name='fc1_Weights')
        fc1_biases = tf.Variable(tf.zeros([1,50]),name='fc1_biases')
    with tf.name_scope('prediction'):
        predictions = tf.add(tf.matmul(l4_flat,fc1_weights),fc1_biases)
    with tf.name_scope('loss'):
        loss = tf.losses.mean_squared_error(predictions,points,weights=1,scope='loss')
        # tf.summary.scalar('loss',loss)
    with tf.name_scope('training'):
        train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

tf.summary.scalar(name='loss',tensor=loss)
batch_size = 100
all_size = 50000
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    tf.global_variables_initializer().run()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(1000):
        begin = epoch*batch_size%(all_size-batch_size)
        end = begin + batch_size
        batch_images = data_images[begin:end]
        print(batch_images.shape)
        batch_points = data_points[epoch*batch_size:(epoch+1)*batch_size]
        train_summary,loss_value=sess.run([train_step,loss],feed_dict={image_inputs:batch_images,points:batch_points})
        if (epoch%10==0):
            train_writer.add_summary(train_summary,epoch)
            print(loss_value)
    train_writer.close()
