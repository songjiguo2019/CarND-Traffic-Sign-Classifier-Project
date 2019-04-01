import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import pickle
from sklearn.utils import shuffle
from scipy import ndimage, misc
import matplotlib.image as mpimg
import cv2
import glob

output_images_path = 'outputimages/'
#plt.rcParams.update({'font.size': 8})

# the path to the data
training_file = '../data/train.p'
validation_file = '../data/valid.p'
testing_file = '../data/test.p'

# generate this once and reuse for model training -- only run once!
transform_pickle_file = 'transformed_data_2.pickle'

savemodel_file = 'traffic_model'

# set iteratation number
EPOCHS = 50
# set batch size
BATCH_SIZE = 128

# placeholder for input data (for arbitrary input data size)
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
# placeholder for labels (for arbitrary size)
y = tf.placeholder(tf.int32, None)
# placeholder for dropout
keep_prob = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)

accuracy_operation = 0

# learning rate
rate = 0.001


# LeNet model (5 layers)  -- return logits
def LeNet(n_classes):
    # weight and bias initial value for each layer
    mu = 0
    sigma = 0.1

    # first layer
    # input image size 32x32x1, output 28x28x6
    # filter size is set to 5x5
    # input depth is 3
    # output depth is 6
    conv1_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6),mean=mu, stddev=sigma))
    # initialize parameter of dimension, mean and deviation
    conv1_b = tf.Variable(tf.zeros(6))
    # use tf.conv2d function to run the convolution, stride:1x1
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, conv1_b)
    # RELU activation
    conv1 = tf.nn.relu(conv1)
    # using max pooling algorithm to down sample
    # input:28x28x6, output:14x14x6, conv kernel:2x2, stride:2x2
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # second layer
    # input:14x14x6, output: 10x10x16
    conv2_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_w, strides=[1, 1, 1, 1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, conv2_b)
    # RELU activation function
    conv2 = tf.nn.relu(conv2)
    # using max pooling algorithm to down sample
    # input:10x10x16, output:5x5x16, conv kernel:2x2, stride:2x2
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # flatten into a 1-d vector
    # input:5x5x16, output:400
    fc0 = flatten(conv2)

    # third layer -- fully connected
    # input:400, output:120
    fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_w)
    fc1 = tf.add(fc1, fc1_b)
    # RELU activation
    fc1 = tf.nn.relu(fc1)
    # dropout
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # fourth layer -- fully connected
    # input:120, output:84
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_w)
    fc2 = tf.add(fc2, fc2_b)
    # RELU activation
    fc2 = tf.nn.relu(fc2)
    # dropout
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # fifth layer -- fully connected
    # input:84, output:10 (the width of layer is same as the number of classes)
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(n_classes))
    # comput logits here
    logits = tf.matmul(fc2, fc3_w) + fc3_b

    return logits


def grayscale_batch(images):
    """ to return gray images"""
    grayimages = np.empty(images.shape[:-1])
    for i in range(0, len(images)):
        grayimages[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)

    return np.expand_dims(grayimages, axis=-1)


# def image_normalize(img):
#     """ normalization function """
#     return (img-128.)/128.


#def image_normalize(image):
#    image_gray = np.mean(image, axis=3)
#    image_gray = np.expand_dims(image_gray, axis=3)
#    image_norm = (image_gray - image_gray.mean()) / image_gray.std()

#    return image_norm

def image_normalize(img):
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean)/std


# input: one of data set (train, test and validation)  -- this needs be called in a session
def evaluate(x_data, y_data):
    num_examples = len(x_data)
    total_accuracy = 0

    sess = tf.get_default_session()
    # start evaluating the data
    for off_set in range(0, num_examples, BATCH_SIZE):
        batchx, batchy = x_data[off_set:off_set + BATCH_SIZE], y_data[off_set:off_set + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batchx, y: batchy, keep_prob: 1.0})
        # get total accuracy
        total_accuracy += (accuracy * len(batchx))
    return total_accuracy / num_examples


def test_model(data, labels):

    with tf.Session() as sess:
        # load the trained model we saved
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('./' + savemodel_file + '.meta')
        saver.restore(sess, './' + savemodel_file)
        test_accuracy = evaluate(data, labels)
        print("Test Mode Fn: Accuracy = {:.3f}".format(test_accuracy))


def prediction():
    fig, axs = plt.subplots(2, 4, figsize=(4, 2))
    fig.subplots_adjust(hspace=.2, wspace=.001)
    axs = axs.ravel()

    my_images = []

    for i, img in enumerate(glob.glob('./test_images/*.png')):
        image = cv2.imread(img)
        axs[i].axis('off')
        axs[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        my_images.append(image)

    new_images = np.array(my_images)
    new_images_gray = grayscale_batch(new_images)
    new_images_gray_normalized = image_normalize(new_images_gray)
    print(new_images_gray_normalized.shape)

    #plt.show()
    plt.savefig(output_images_path + 'prediction_test_images.png', bbox_inches='tight')

    test_labels = [14, 11, 1, 12, 38, 18, 25, 34]
    test_model(new_images_gray_normalized, test_labels)

    return new_images_gray_normalized, new_images


# train and validate our model
def train_model(n_train, training_operation, x_valid_normalized, y_valid,
                x_train_normalized, y_train, x_test_normalized, y_test):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Training...")
        print()

        for i in range(EPOCHS):
            # randomize the train data to minimize the impact from the data order  -- otherwise, the accuracy is low!!
            x_train_normalized, y_train = shuffle(x_train_normalized, y_train)
            # train the mode in multiple batch

            for offset in range(0, n_train, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = x_train_normalized[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.8})
            # check the accuracy after each train using train data
            train_accuracy = evaluate(x_train_normalized, y_train)
            # check the accuracy after each train using validation data
            validation_accuracy = evaluate(x_valid_normalized, y_valid)
            # check the accuracy after each train using test data
            test_accuracy = evaluate(x_test_normalized, y_test)
            print("EPOCH {}...".format(i + 1))
            print("Train accuracy = {:.3f}".format(train_accuracy))
            print("Validation accuracy = {: .3f}".format(validation_accuracy))
            print("Test accuracy = {: .3f}".format(test_accuracy))
            print()

        # save the model
        saver = tf.train.Saver()
        saver.save(sess, savemodel_file)
        print("Model saved in %s" % savemodel_file)


def train_init(n_classes):
    global accuracy_operation
    # using one_hot encoding
    one_hot_y = tf.one_hot(y, n_classes)

    # get logits
    logits = LeNet(n_classes)
    # classification function -- compute corss entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    # compute the average of all cross_entropy
    loss_operation = tf.reduce_mean(cross_entropy)
    # optimize with Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    # minimize the loss
    training_operation = optimizer.minimize(loss_operation)

    # tf.argmax to get the max index, tf.equal to indicate true or false
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    # compute overall accuracy which is the average of all prediction
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return logits, training_operation


def transform_image(img, ang_range, shear_range, trans_range):
    """
    transforms images to generate new images by
     -- ang_range: Range of angles for rotation
     -- shear_range: Range of values to apply affine transform to
     -- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation
    """

    # Rotation
    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    rows, cols, ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

    shear_M = cv2.getAffineTransform(pts1, pts2)

    img = cv2.warpAffine(img, Rot_M, (cols, rows))
    img = cv2.warpAffine(img, Trans_M, (cols, rows))
    img = cv2.warpAffine(img, shear_M, (cols, rows))

    return img


def transform_training_data(x_train, y_train):
    # Example image Without transformation
    image_num = 3000
    image = x_train[image_num, :, :, :]
    image = transform_image(image, 0, 0, 0)
    plt.imshow(image)
    #plt.show()
    #print("Image data shape =", image_shape)
    plt.savefig(output_images_path + 'orignal_image.png', bbox_inches='tight')

    # Example image with transformation
    image_num = 3000
    image = x_train[image_num, :, :, :]
    image = transform_image(image, 10, 10, 10)
    plt.imshow(image)
    #plt.show()
    #print("Image data shape =", image_shape)
    plt.savefig(output_images_path + 'transformed_image.png', bbox_inches='tight')

    values, counts = np.unique(y_train, return_counts=True)
    max_counts = counts.max()

    for class_ in values:
        print('working on class #', class_, '...')
        num_img_needed = max_counts-counts[class_]
        first_example_index = next(index for index, val in enumerate(y_train) if val==class_)
        first_example = x_train[first_example_index]
        class_ = class_.reshape([1])

        for num in range(0, num_img_needed):
            transformed_example = transform_image(first_example, 5, 5, 5)
            transformed_example = transformed_example.reshape([1, 32, 32, 3])
            x_train = np.concatenate([x_train, transformed_example])
            y_train = np.concatenate([y_train, class_])

    # do this only once
    if not os.path.isfile(transform_pickle_file):
        print('Saving data to pickle file...')
        try:
            with open('transformed_data_2.pickle', 'wb') as pfile:
                pickle.dump(
                    {
                        'X_train': x_train,
                        'y_train': y_train,
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', transform_pickle_file, ':', e)
            raise

    print('Data cached in pickle file %s' % transform_pickle_file)


def load_transformed_traning_data():
    # Reload the data
    with open(transform_pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        x_train = pickle_data['X_train']
        y_train = pickle_data['y_train']
        # del pickle_data  # Free up memory

    return x_train, y_train


def pre_process(x_train, y_train, x_valid, y_valid, x_test, y_test, n_train,
                n_validation, n_test, n_classes):

    index = random.randint(0, len(x_train))
    image = x_train[index].squeeze()
    plt.figure(figsize=(1, 1))
    plt.imshow(image, cmap="gray")
    #plt.show()
    plt.savefig(output_images_path + 'random_train_image_preprocess.png', bbox_inches='tight')

    # Converting to grayscale images
    x_train = grayscale_batch(x_train)
    x_valid = grayscale_batch(x_valid)
    x_test = grayscale_batch(x_test)

    image = x_train[index].squeeze()
    plt.figure(figsize=(1, 1))
    plt.imshow(image, cmap="gray")
    #plt.show()
    plt.savefig(output_images_path + 'grayscaled_train_image_preprocess.png', bbox_inches='tight')

    # normalizing images
    x_train = image_normalize(x_train)
    x_valid = image_normalize(x_valid)
    x_test = image_normalize(x_test)

    image = x_train[index].squeeze()
    plt.figure(figsize=(1, 1))
    plt.imshow(image, cmap="gray")
    #plt.show()
    plt.savefig(output_images_path + 'normalized_train_image_preprocess.png', bbox_inches='tight')


    #return x_train_normalized, x_valid_normalized, x_test_normalized
    return x_train, x_valid, x_test


def init():
    # loading the data
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    x_train, y_train = train['features'], train['labels']
    # do this only once
    #transform_training_data(x_train, y_train)

    x_valid, y_valid = valid['features'], valid['labels']
    x_test, y_test = test['features'], test['labels']

    # show the number of each classes before transforming the data
    cmbin = [x for x in range(0, len(np.unique(y_train)))]
    plt.ylabel('Count of Examples')
    plt.xlabel('Traffic Sign Number')
    plt.hist(y_train, cmbin, color='r', rwidth=0.8, label='train')
    plt.hist(y_test, cmbin, color='c', rwidth=0.8, label='test')
    plt.hist(y_valid, cmbin, color='b', rwidth=0.8, label='valid')
    plt.legend()
    #plt.show()
    plt.savefig(output_images_path + 'number_of_classes.png', bbox_inches='tight')

    x_train, y_train = load_transformed_traning_data()  # load train data/labels from transformed data (better accuracy)



    # show the number of each classes before transforming the data
    cmbin = [x for x in range(0, len(np.unique(y_train)))]
    plt.ylabel('Count of Examples')
    plt.xlabel('Traffic Sign Number')
    plt.hist(y_train, cmbin, color='r', rwidth=0.8, label='train')
    plt.hist(y_test, cmbin, color='c', rwidth=0.8, label='test')
    plt.hist(y_valid, cmbin, color='b', rwidth=0.8, label='valid')
    plt.legend()
    #plt.show()
    plt.savefig(output_images_path + 'normalized_classes.png', bbox_inches='tight')


    # train data size
    n_train = len(x_train)
    # validation data size
    n_validation = len(x_valid)
    # test data size
    n_test = len(x_test)
    # image size
    image_shape = x_train.shape
    # number of image classes
    n_classes = len(np.unique(y_train))

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Number of validation examples =", n_validation)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    # use some random index and get the image from train data set
    # just view some image to make sure everything is right
    index = random.randint(0, len(x_train))
    image = x_train[index].squeeze()
    plt.figure(figsize=(1, 1))
    plt.imshow(image, cmap="gray")
    #plt.show()
    plt.savefig(output_images_path + 'random_train_image.png', bbox_inches='tight')
    #print(y_train[index])

    return x_train, y_train, x_valid, y_valid, x_test, y_test, n_train, n_validation, n_test, n_classes


def top_five(logits, my_images_normalized, images, x_valid, y_valid):
    softmax_logits = tf.nn.softmax(logits)
    top_k = tf.nn.top_k(softmax_logits, k=5)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('./' + savemodel_file + '.meta')
        saver.restore(sess, './' + savemodel_file)
        my_softmax_logits = sess.run(softmax_logits,
                                     feed_dict={x: my_images_normalized, keep_prob: 1.0, keep_prob2: 1.0})
        my_top_k = sess.run(top_k, feed_dict={x: my_images_normalized, keep_prob: 1.0, keep_prob2: 1.0})

        fig, axs = plt.subplots(len(images), 6, figsize=(12, 14))
        fig.subplots_adjust(hspace=.4, wspace=.2)
        axs = axs.ravel()

        for i, image in enumerate(images):
            axs[6 * i].axis('off')
            axs[6 * i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axs[6 * i].set_title('input')
            guess1 = my_top_k[1][i][0]
            index1 = np.argwhere(y_valid == guess1)[0]
            axs[6 * i + 1].axis('off')
            axs[6 * i + 1].imshow(x_valid[index1].squeeze(), cmap='gray')
            axs[6 * i + 1].set_title('top guess: {} ({:.0f}%)'.format(guess1, 100 * my_top_k[0][i][0]))
            guess2 = my_top_k[1][i][1]
            index2 = np.argwhere(y_valid == guess2)[0]
            axs[6 * i + 2].axis('off')
            axs[6 * i + 2].imshow(x_valid[index2].squeeze(), cmap='gray')
            axs[6 * i + 2].set_title('2nd guess: {} ({:.0f}%)'.format(guess2, 100 * my_top_k[0][i][1]))
            guess3 = my_top_k[1][i][2]
            index3 = np.argwhere(y_valid == guess3)[0]
            axs[6 * i + 3].axis('off')
            axs[6 * i + 3].imshow(x_valid[index3].squeeze(), cmap='gray')
            axs[6 * i + 3].set_title('3rd guess: {} ({:.0f}%)'.format(guess3, 100 * my_top_k[0][i][2]))

            guess4 = my_top_k[1][i][3]
            index4 = np.argwhere(y_valid == guess4)[0]
            axs[6 * i + 4].axis('off')
            axs[6 * i + 4].imshow(x_valid[index4].squeeze(), cmap='gray')
            axs[6 * i + 4].set_title('4th guess: {} ({:.0f}%)'.format(guess4, 100 * my_top_k[0][i][3]))

            guess5 = my_top_k[1][i][4]
            index5 = np.argwhere(y_valid == guess5)[0]
            axs[6 * i + 5].axis('off')
            axs[6 * i + 5].imshow(x_valid[index5].squeeze(), cmap='gray')
            axs[6 * i + 5].set_title('5th guess: {} ({:.0f}%)'.format(guess5, 100 * my_top_k[0][i][4]))

        #plt.show()
        plt.savefig(output_images_path + 'top_five.png', bbox_inches='tight')


def main():

    x_train, y_train, x_valid, y_valid, x_test, y_test, n_train, n_validation, n_test, n_classes = init()

    x_train_normalized, x_valid_normalized, x_test_normalized = pre_process(x_train, y_train, x_valid, y_valid, x_test,
                                                                            y_test, n_train, n_validation, n_test,
                                                                            n_classes)

    logits, training_operation = train_init(n_classes)

    train_model(n_train, training_operation, x_valid_normalized, y_valid,
                x_train_normalized, y_train, x_test_normalized, y_test)

    # test_model(x_train_normalized, y_train)
    # test_model(x_valid_normalized, y_valid)
    # test_model(x_test_normalized, y_test)

    new_images_gray_normalized, images = prediction()

    top_five(logits, new_images_gray_normalized, images, x_valid, y_valid)


main()
