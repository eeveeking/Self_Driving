import tensorflow as tf
import numpy as np
import os
import imageio
from glob import glob
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt


DEBUG = 1
TRAIN_DATA_SIZE = 20
TEST_DATA_SIZE = 2000
BATCH_SIZE = 4
SAVE_PATH = './model/m.cpkt'
RESTORE = True


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

def data_load(image_files, bbox_files, ALL_IMAGE, ALL_BBOX):
    print("Begin loading data...")
    if DEBUG:
        image_files = image_files[:TRAIN_DATA_SIZE]
    for file in image_files:
        # Read images
        img = plt.imread(file)
        ALL_IMAGE.append(img)

        # Read bbox
        try:
            bbox = np.fromfile(file.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
            bbox = bbox[:9]
            bbox = bbox.reshape([-1, 9])
            ALL_BBOX.append(bbox)
        except FileNotFoundError:
            print('[*] bbox not found.')
            bbox = np.array([], dtype=np.float32)
        # print(bbox)

    ALL_IMAGE = np.reshape(np.array(ALL_IMAGE), (TRAIN_DATA_SIZE, 1052, 1914, 1))
    print(ALL_IMAGE.shape)
    ALL_BBOX = np.array(ALL_BBOX)
    print(ALL_BBOX.shape)

    return ALL_IMAGE, ALL_BBOX



def normalize(ALL_IMAGE, ALL_MASK, ALL_NORMAL):
    print("begin normalizing data")
    ALL_IMAGE = ALL_IMAGE / 255


def unet(X_train):
    conv1 = tf.layers.conv2d(
        inputs=X_train,
        filters=64,
        kernel_size=[3,3],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu)

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=3,
        kernel_size=[3,3],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu)

    dense1 = tf.layers.dense(
        inputs = conv2,
        units = 9,
        activation=tf.nn.relu,
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

    # Add a linear layer and let the output to be 1*9
    return dense1

#TODO
def find_loss(pred_bbox, true_bbox):
    # take care of mask
    loss = tf.losses.mean_squared_error(pred_bbox, true_bbox)
    return loss

def optimizer(loss):
    # return a tf operation
    return tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

#TODO
def train_cnn(
    sess, saver, images, bbox, loss, train_op, ALL_IMAGE, ALL_BBOX):

    iterations = int(TRAIN_DATA_SIZE/BATCH_SIZE)

    for batch_index in range(iterations):
        batch_images = ALL_IMAGE[BATCH_SIZE*(batch_index):BATCH_SIZE*(batch_index + 1)]
        batch_bbox = ALL_BBOX[BATCH_SIZE*(batch_index):BATCH_SIZE*(batch_index + 1)]
        sess.run(train_op, feed_dict={images: batch_images, bbox: batch_bbox})
        cur_loss = sess.run(loss, feed_dict={images: batch_images, bbox: batch_bbox})
        print("loss for batch {} is {}".format(batch_index, cur_loss))

    saver.save(sess, SAVE_PATH)

# def train_unet():


def prediction(t_images, t_mask, pred_model):
    test_image_folder = './test/color/'
    test_mask_folder = './test/mask/'
    test_normal_folder = './test/normal/'
    test_image = []
    test_mask = []
    print("begin loading test data")
    file_list = os.listdir(test_image_folder)
    for name in file_list:
        img = imread(os.path.join(test_image_folder, name))[:,:,0]
        mask = imread(os.path.join(test_mask_folder, name))
        test_image.append(img)
        test_mask.append(mask)
    test_image = np.reshape(np.array(test_image, dtype='f'), (TEST_DATA_SIZE, 128, 128, 1))
    test_mask = np.reshape(np.array(test_mask, dtype='f'), (TEST_DATA_SIZE, 128, 128, 1))
    test_image = test_image / 255
    test_mask = test_mask / 255
    print("begin predicting normal")
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, SAVE_PATH)
        print(type(test_image[0,0,0,0]))
        pred_normal = sess.run(pred_model, feed_dict={t_images: test_image, t_mask: test_mask})

    for idx in range(TEST_DATA_SIZE):
        img = pred_normal[idx,:,:,:]
        img = ((img/2.0+0.5)*255).round()
        # img = img * test_mask[idx,:,:,:]
        img = img.astype(np.uint8)
        filename = str(idx) + '.png'
        imageio.imwrite(test_normal_folder + filename, img)



def main():
    image_files = glob('deploy/trainval/*/*_image.jpg')
    bbox_files = glob('deploy/trainval/*/*_bbox.bin')
    # Image_folder = './train/color/'
    # Mask_folder = './train/mask/'
    # Normal_folder = './train/normal/'

    ALL_IMAGE = []
    ALL_BBOX = []
    print('building model...')
    ALL_IMAGE, ALL_BBOX = data_load(image_files, bbox_files, ALL_IMAGE, ALL_BBOX)
    #Not sure if we should normalize
    normalize(ALL_IMAGE, ALL_BBOX)
    # placeholders
    images = tf.placeholder(tf.float32, [None, 1052, 1914, 3])
    bbox = tf.placeholder(tf.float32, [None, 1, 9])
    pred_model = unet(images)
    #calculate loss
    loss = find_loss(pred_model, bbox)
    # optimizer
    train_op = optimizer(loss)
    saver = tf.train.Saver()
    # train
    with tf.Session() as sess:
        if RESTORE:
            sess.run(tf.global_variables_initializer())
        #TODO
        train_cnn(sess, saver, images, bbox, loss, train_op, ALL_IMAGE, ALL_BBOX)

    t_images = tf.placeholder(tf.float32, [None, 1052, 1914, 3])
    #TODO
    prediction(images, pred_model)



if __name__ == '__main__':
    main()
