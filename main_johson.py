import tensorflow as tf
import numpy as np
import os
from glob import glob
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import cv2
from random import randint
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model

DEBUG = 1
if DEBUG:
    TRAIN_DATA_SIZE = 200
    TEST_DATA_SIZE = 200
    epoch = 10
else:
    TRAIN_DATA_SIZE = 7573
    TEST_DATA_SIZE = 2631
    epoch = 50
BATCH_SIZE = 128
SAVE_PATH = './model/m.cpkt'
RESTORE = True
IMGSIZE = 224


# def data_load(Image_folder, Mask_folder, Normal_folder, ALL_IMAGE, ALL_MASK, ALL_NORMAL):
#     print("begin loading data")
#     image_list = os.listdir(Image_folder)[0:TRAIN_DATA_SIZE]
#     for name in image_list:
#         img = imread(os.path.join(Image_folder, name))[:,:,0]
#         mask = imread(os.path.join(Mask_folder, name))
#         normal = imread(os.path.join(Normal_folder, name))
#         ALL_IMAGE.append(img)
#         ALL_MASK.append(mask)
#         ALL_NORMAL.append(normal)
#     ALL_IMAGE = np.reshape(np.array(ALL_IMAGE), (TRAIN_DATA_SIZE, 128, 128, 1))
#     ALL_MASK = np.reshape(np.array(ALL_MASK), (TRAIN_DATA_SIZE, 128, 128, 1))
#     ALL_NORMAL = np.array(ALL_NORMAL)
#     print("shape of container!")
#     print(type(ALL_IMAGE))
#     print(ALL_MASK.shape)
#     print(ALL_NORMAL.shape)
#     return ALL_IMAGE, ALL_MASK, ALL_NORMAL

def data_load(image_files, bbox_files, ALL_IMAGE, ALL_BBOX, base_model):
    print("Begin loading data...")
    if DEBUG:
        image_files = image_files[:TRAIN_DATA_SIZE]
    for file in image_files:
        # Read images
        img = cv2.imread(file)
        img = cv2.resize(img,(IMGSIZE,IMGSIZE)).astype(np.float32)
        img -= [103.939, 116.779, 123.68]
        img = img / 255
        img = load_pretrained_vgg19_fc2(img, base_model)
        ALL_IMAGE.append(img)

        # Read bbox
        try:
            bbox = np.fromfile(file.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
            bbox = bbox[:9]
            # bbox = bbox.reshape([-1, 9])
            ALL_BBOX.append(bbox)
        except FileNotFoundError:
            print('[*] bbox not found.')
            bbox = np.array([], dtype=np.float32)
        # print(bbox)

    ALL_IMAGE = np.reshape(np.array(ALL_IMAGE), (TRAIN_DATA_SIZE, 1, 4096))
    print(ALL_IMAGE.shape)
    ALL_BBOX = np.array(ALL_BBOX)
    print(ALL_BBOX.shape)

    return ALL_IMAGE, ALL_BBOX

def normalize(x):
    ''' Set mean to 0.0 and standard deviation to 1.0 via affine transform '''
    shifted = x - tf.reduce_mean(x)
    scaled = shifted / tf.sqrt(tf.reduce_mean(tf.multiply(shifted, shifted)))
    return scaled

# def normalize(ALL_IMAGE):
#     print("begin normalizing data")
#     ALL_IMAGE = ALL_IMAGE / 255


def vgg16net(X_train):
    # conv1 = tf.layers.conv2d(
    #     inputs=X_train,
    #     filters=64,
    #     kernel_size=[3,3],
    #     strides=(2, 2),
    #     padding='same',
    #     activation=tf.nn.relu)

    # conv2 = tf.layers.conv2d(
    #     inputs=conv1,
    #     filters=64,
    #     kernel_size=[3,3],
    #     strides=(2, 2),
    #     padding='same',
    #     activation=tf.nn.relu)

    # maxpool1 = tf.nn.max_pool(
    #     conv2, 
    #     ksize=[1, 2, 2, 1], 
    #     strides=[1, 2, 2, 1], 
    #     padding='SAME')

    # conv3 = tf.layers.conv2d(
    #     inputs=maxpool1,
    #     filters=128,
    #     kernel_size=[3,3],
    #     strides=(2, 2),
    #     padding='same',
    #     activation=tf.nn.relu)

    # conv4 = tf.layers.conv2d(
    #     inputs=conv3,
    #     filters=128,
    #     kernel_size=[3,3],
    #     strides=(2, 2),
    #     padding='same',
    #     activation=tf.nn.relu)

    # maxpool2 = tf.nn.max_pool(
    #     conv4, 
    #     ksize=[1, 2, 2, 1], 
    #     strides=[1, 2, 2, 1], 
    #     padding='SAME')

    # conv5 = tf.layers.conv2d(
    #     inputs=maxpool2,
    #     filters=256,
    #     kernel_size=[3,3],
    #     strides=(2, 2),
    #     padding='same',
    #     activation=tf.nn.relu)

    # conv6 = tf.layers.conv2d(
    #     inputs=conv5,
    #     filters=256,
    #     kernel_size=[3,3],
    #     strides=(2, 2),
    #     padding='same',
    #     activation=tf.nn.relu)

    # conv7 = tf.layers.conv2d(
    #     inputs=conv6,
    #     filters=256,
    #     kernel_size=[1,1],
    #     strides=(1, 1),
    #     padding='same',
    #     activation=tf.nn.relu)

    # maxpool3 = tf.nn.max_pool(
    #     conv7, 
    #     ksize=[1, 2, 2, 1], 
    #     strides=[1, 2, 2, 1], 
    #     padding='SAME')

    # conv8 = tf.layers.conv2d(
    #     inputs=maxpool3,
    #     filters=512,
    #     kernel_size=[3,3],
    #     strides=(2, 2),
    #     padding='same',
    #     activation=tf.nn.relu)

    # conv9 = tf.layers.conv2d(
    #     inputs=conv8,
    #     filters=512,
    #     kernel_size=[3,3],
    #     strides=(2, 2),
    #     padding='same',
    #     activation=tf.nn.relu)

    # conv10 = tf.layers.conv2d(
    #     inputs=conv9,
    #     filters=512,
    #     kernel_size=[1,1],
    #     strides=(1, 1),
    #     padding='same',
    #     activation=tf.nn.relu)

    # maxpool4 = tf.nn.max_pool(
    #     conv10, 
    #     ksize=[1, 2, 2, 1], 
    #     strides=[1, 2, 2, 1], 
    #     padding='SAME')

    # conv11 = tf.layers.conv2d(
    #     inputs=maxpool4,
    #     filters=512,
    #     kernel_size=[3,3],
    #     strides=(2, 2),
    #     padding='same',
    #     activation=tf.nn.relu)

    # conv12 = tf.layers.conv2d(
    #     inputs=conv11,
    #     filters=512,
    #     kernel_size=[3,3],
    #     strides=(2, 2),
    #     padding='same',
    #     activation=tf.nn.relu)

    # conv13 = tf.layers.conv2d(
    #     inputs=conv12,
    #     filters=512,
    #     kernel_size=[1,1],
    #     strides=(1, 1),
    #     padding='same',
    #     activation=tf.nn.relu)

    # maxpool5 = tf.nn.max_pool(
    #     conv13, 
    #     ksize=[1, 2, 2, 1], 
    #     strides=[1, 2, 2, 1], 
    #     padding='SAME')

    flatten = tf.layers.flatten(
        inputs=X_train)

    dense1 = tf.layers.dense(
        inputs = flatten,
        units = 4096,
        activation=tf.nn.relu
        # use_bias=True,
        # kernel_initializer=None,
        # bias_initializer=tf.zeros_initializer(),
        # kernel_regularizer=None,
        # bias_regularizer=None,
        # activity_regularizer=None,
        # kernel_constraint=None,
        # bias_constraint=None,
        # trainable=True,
        # name=None,
        # reuse=None
    )

    dense2 = tf.layers.dense(
        inputs = dense1,
        units = 1000,
        activation=tf.nn.relu
    )

    dense3 = tf.layers.dense(
        inputs = dense2,
        units = 3,
        activation=tf.nn.softmax
    )

    return dense3

#TODO
def find_loss(pred_bbox, true_bbox):
    # take care of mask
    loss = tf.losses.mean_squared_error(pred_bbox, true_bbox)
    return loss

def optimizer(loss):
    # return a tf operation
    return tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

# def train_cnn(
#     sess, saver, images, bbox, loss, train_op, ALL_IMAGE, ALL_BBOX):

#     iterations = int(TRAIN_DATA_SIZE/BATCH_SIZE)
#     shuffle_index = list(range(TRAIN_DATA_SIZE))
#     np.random.shuffle(shuffle_index)

#     for batch_index in range(iterations):
#         batch_images = [ALL_IMAGE[i] for i in shuffle_index[BATCH_SIZE*(batch_index):BATCH_SIZE*(batch_index + 1)]]
#         # batch_images = ALL_IMAGE[BATCH_SIZE*(batch_index):BATCH_SIZE*(batch_index + 1)]
#         batch_bbox = [ALL_BBOX[i] for i in shuffle_index[BATCH_SIZE*(batch_index):BATCH_SIZE*(batch_index + 1)]]
#         # batch_bbox = ALL_BBOX[BATCH_SIZE*(batch_index):BATCH_SIZE*(batch_index + 1)]
#         _, cur_loss = sess.run([train_op, loss], feed_dict={images: batch_images, bbox: batch_bbox})
#         # cur_loss = sess.run(loss, feed_dict={images: batch_images, bbox: batch_bbox})
#         print("loss for batch {} is {}".format(batch_index, cur_loss))

#     # saver.save(sess, SAVE_PATH)
#     return sess


def train_cnn(
    sess, saver, images, labels, loss, train_op, ALL_IMAGE, ALL_LABEL):

    iterations = int(TRAIN_DATA_SIZE/BATCH_SIZE) + 1
    shuffle_index = list(range(TRAIN_DATA_SIZE))
    np.random.shuffle(shuffle_index)

    for batch_index in range(iterations):
        batch_images = [ALL_IMAGE[i] for i in shuffle_index[BATCH_SIZE*(batch_index):BATCH_SIZE*(batch_index + 1)]]
        # batch_images = ALL_IMAGE[BATCH_SIZE*(batch_index):BATCH_SIZE*(batch_index + 1)]
        batch_labels = [ALL_LABEL[i] for i in shuffle_index[BATCH_SIZE*(batch_index):BATCH_SIZE*(batch_index + 1)]]
        # batch_labels = ALL_LABEL[BATCH_SIZE*(batch_index):BATCH_SIZE*(batch_index + 1)]
        _, cur_loss = sess.run([train_op, loss], feed_dict={images: batch_images, labels: batch_labels})
        # cur_loss = sess.run(loss, feed_dict={images: batch_images, bbox: batch_bbox})
        print("loss for batch {} is {}".format(batch_index, cur_loss))
    
    # saver.save(sess, SAVE_PATH)
    
    # tf.saved_model.simple_save(sess, SAVE_PATH)
    return sess


def prediction(images, pred_model, sess, base_model):
    test_image_files = glob('deploy/test/*/*_image.jpg')
    if DEBUG:
        test_image_files = test_image_files[:TEST_DATA_SIZE]
    TEST_IMAGE = []
    print("begin loading test data")

    for file in test_image_files:
        # Read images
        img = cv2.imread(file)
        img = cv2.resize(img,(IMGSIZE,IMGSIZE)).astype(np.float32)
        img -= [103.939, 116.779, 123.68]
        img = img / 255
        img = load_pretrained_vgg19_fc2(img, base_model)
        TEST_IMAGE.append(img)

    TEST_IMAGE = np.reshape(np.array(TEST_IMAGE), (TEST_DATA_SIZE, 1, 4096))
    # normalize(TEST_IMAGE)

    print(TEST_IMAGE.shape)
    iterations = int(TEST_DATA_SIZE/BATCH_SIZE) + 1


    print("begin predicting normal")
    pred_labels = np.array([])
    # with tf.Session() as sess:
    #     # sess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver()
        # saver.restore(sess, SAVE_PATH)
    for batch_index in range(iterations):
        batch_images = TEST_IMAGE[BATCH_SIZE*(batch_index):BATCH_SIZE*(batch_index + 1)]
        batch_labels = sess.run(pred_model, feed_dict={images: batch_images})
        pred_labels = np.append(pred_labels, batch_labels)

    # print(pred_labels)
    return pred_labels


def load_pretrained_vgg19_fc2(img, base_model):
    # pre-process the image
    # img = image.load_img('./data/peacock.jpg', target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # define model from base model for feature extraction from fc2 layer
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
    # obtain the outpur of fc2 layer
    fc2_features = model.predict(img)

    return fc2_features


def main():
    image_files = glob('deploy/trainval/*/*_image.jpg')
    bbox_files = glob('deploy/trainval/*/*_bbox.bin')
    test_image_files = glob('deploy/test/*/*_image.jpg')
    # Image_folder = './train/color/'
    # Mask_folder = './train/mask/'
    # Normal_folder = './train/normal/'

    # load pre-trained model
    base_model = VGG19(weights='imagenet')

    ALL_IMAGE = []
    ALL_BBOX = []

    # Read labels
    with open('labels.csv', 'r') as f:
        lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]
    ALL_LABEL = []
    for line in lines[1:]:
        TEMP_LABEL = [0,0,0]
        TEMP_LABEL[int(line.split(',')[1])] = 1
        # TEMP_LABEL[randint(0,2)] = 1
        ALL_LABEL.append(TEMP_LABEL)
    ALL_LABEL = np.array(ALL_LABEL)
    if DEBUG:
        ALL_LABEL = ALL_LABEL[:TRAIN_DATA_SIZE]
    print(ALL_LABEL.shape)

    print('building model...')
    ALL_IMAGE, ALL_BBOX = data_load(image_files, bbox_files, ALL_IMAGE, ALL_BBOX, base_model)
    #Not sure if we should normalize
    # normalize(ALL_IMAGE)
    # placeholders
    images = tf.placeholder(tf.float32, [None, 1, 4096])
    bbox = tf.placeholder(tf.float32, [None, 9])
    labels = tf.placeholder(tf.float32, [None, 3])

    pred_model = vgg16net(images)
    #calculate loss
    loss = find_loss(pred_model, labels)
    # optimizer
    train_op = optimizer(loss)
    saver = tf.train.Saver()
    # train
    with tf.Session() as sess:
        if RESTORE:
            sess.run(tf.global_variables_initializer())
        #TODO
        # for epoch in range(10):
        #     train_cnn(sess, saver, images, bbox, loss, train_op, ALL_IMAGE, ALL_BBOX)
        for i in range(epoch):
            train_cnn(sess, saver, images, labels, loss, train_op, ALL_IMAGE, ALL_LABEL)

        # t_images = tf.placeholder(tf.float32, [None, IMGSIZE, IMGSIZE, 3])
    #TODO
        pred_labels = prediction(images, pred_model, sess, base_model)

    # print(pred_labels.shape)
    label_output = ''
    for i in range(TEST_DATA_SIZE):
        score = pred_labels[3*i:3*i+3]
        # print(score)
        idx = np.argmax(score)
        label_output += str(test_image_files[i])
        label_output += ',' + str(idx) + '\n'

    with open('output.csv', 'w') as f:
        f.write(label_output)



if __name__ == '__main__':
    main()
