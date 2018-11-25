import tensorflow as tf
import numpy as np
import os
import imageio
from scipy.misc import imread, imresize

TRAIN_DATA_SIZE = 20000
TEST_DATA_SIZE = 2000
BATCH_SIZE = 128
SAVE_PATH = './model/m.cpkt'
RESTORE = True

def data_load(Image_folder, Mask_folder, Normal_folder, ALL_IMAGE, ALL_MASK, ALL_NORMAL):
    print("begin loading data")
    image_list = os.listdir(Image_folder)[0:TRAIN_DATA_SIZE]
    for name in image_list:
        img = imread(os.path.join(Image_folder, name))[:,:,0]
        mask = imread(os.path.join(Mask_folder, name))
        normal = imread(os.path.join(Normal_folder, name))
        ALL_IMAGE.append(img)
        ALL_MASK.append(mask)
        ALL_NORMAL.append(normal)
    ALL_IMAGE = np.reshape(np.array(ALL_IMAGE), (TRAIN_DATA_SIZE, 128, 128, 1))
    ALL_MASK = np.reshape(np.array(ALL_MASK), (TRAIN_DATA_SIZE, 128, 128, 1))
    ALL_NORMAL = np.array(ALL_NORMAL)
    print("shape of container!")
    print(type(ALL_IMAGE))
    print(ALL_MASK.shape)
    print(ALL_NORMAL.shape)
    return ALL_IMAGE, ALL_MASK, ALL_NORMAL

def normalize(ALL_IMAGE, ALL_MASK, ALL_NORMAL):
    print("begin normalizing data")
    ALL_IMAGE = ALL_IMAGE / 255
    ALL_NORMAL = ((ALL_NORMAL / 255) - 0.5) * 2 # -1 to 1
    ALL_MASK = ALL_MASK / 255


def unet(X_train):
    conv1 = conv1 = tf.layers.conv2d(
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
    return conv2


def find_loss(pred_normal, mask, true_normal):
    # take care of mask
    loss = tf.losses.mean_squared_error(tf.multiply(true_normal, mask), tf.multiply(pred_normal, mask))
    return loss

def optimizer(loss):
    # return a tf operation
    return tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

def train_cnn(
    sess, saver, images, mask, normal, loss, train_op, ALL_IMAGE, ALL_MASK, ALL_NORMAL):
    
    iterations = int(TRAIN_DATA_SIZE/BATCH_SIZE)

    for batch_index in range(iterations):
        batch_images = ALL_IMAGE[BATCH_SIZE*(batch_index):BATCH_SIZE*(batch_index + 1)]
        batch_normal = ALL_NORMAL[BATCH_SIZE*(batch_index):BATCH_SIZE*(batch_index + 1)]
        batch_mask = ALL_MASK[BATCH_SIZE*(batch_index):BATCH_SIZE*(batch_index + 1)]
        sess.run(train_op, feed_dict={images: batch_images, mask: batch_mask, normal: batch_normal})
        cur_loss = sess.run(loss, feed_dict={images: batch_images, mask: batch_mask, normal: batch_normal})
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
    Image_folder = './train/color/'
    Mask_folder = './train/mask/'
    Normal_folder = './train/normal/'
    
    ALL_IMAGE = []
    ALL_MASK = []
    ALL_NORMAL = []
    print('building model...')
    ALL_IMAGE, ALL_MASK, ALL_NORMAL = data_load(Image_folder, Mask_folder, Normal_folder, ALL_IMAGE, ALL_MASK, ALL_NORMAL)
    normalize(ALL_IMAGE, ALL_MASK, ALL_NORMAL)
    # placeholders
    images = tf.placeholder(tf.float32, [None, 128, 128, 1])
    mask = tf.placeholder(tf.float32, [None, 128, 128, 1])
    normal = tf.placeholder(tf.float32, [None, 128, 128, 3])
    pred_model = unet(images)
    #calculate loss
    loss = find_loss(pred_model, mask, normal)
    # optimizer
    train_op = optimizer(loss)
    saver = tf.train.Saver()
    # train
    with tf.Session() as sess:
        if RESTORE:
            sess.run(tf.global_variables_initializer())
        train_cnn(sess, saver, images, mask, normal, loss, train_op, ALL_IMAGE, ALL_MASK, ALL_NORMAL)

    t_images = tf.placeholder(tf.float32, [None, 128, 128, 1])
    t_mask = tf.placeholder(tf.float32, [None, 128, 128, 1])
    prediction(images, mask, pred_model)



if __name__ == '__main__':
    main()
