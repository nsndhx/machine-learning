# coding=utf-8
import sys
import time
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt


# Number of epochs & learning rate in the original paper
epochs_original, lr_global_original = 16, np.array([5e-4] * 2 + [2e-4] * 3 + [1e-4] * 3 + [5e-5] * 4 + [1e-5] * 8)
# Number of epochs & learning rate I used
epochs, lr_global_list = epochs_original, lr_global_original * 100


def train(LeNet5, train_images, train_labels):
    momentum = 0.9
    weight_decay = 0
    batch_size = 256

    # Training loops
    cost_last, count = np.Inf, 0
    err_rate_list = []
    for epoch in range(0, epochs):
        print("----------------------------------- epoch{} begin -----------------------------------".format(epoch + 1))

        # Stochastic Diagonal Levenberg-Marquardt method for determining the learning rate
        batch_image, batch_label = random_mini_batches(train_images, train_labels, mini_batch_size=500, one_batch=True)
        LeNet5.Forward_Propagation(batch_image, batch_label, 'train')
        lr_global = lr_global_list[epoch]
        LeNet5.SDLM(0.02, lr_global)

        # print info
        print("global learning rate:", lr_global)
        print("learning rates in trainable layers:", np.array([LeNet5.C1.lr, LeNet5.C3.lr, LeNet5.C5.lr, LeNet5.F6.lr]))
        print("batch size:", batch_size)
        print("momentum:", momentum, ", weight decay:", weight_decay)

        # loop over each batch
        ste = time.time()
        cost = 0
        mini_batches = random_mini_batches(train_images, train_labels, batch_size)
        for i in range(len(mini_batches)):
            batch_image, batch_label = mini_batches[i]

            loss = LeNet5.Forward_Propagation(batch_image, batch_label, 'train')
            cost += loss

            LeNet5.Back_Propagation(momentum, weight_decay)

            # print progress
            if i % (int(len(mini_batches) / 100)) == 0:
                #sys.stdout.write("\033[F")  # CURSOR_UP_ONE
                #sys.stdout.write("\033[K")  # ERASE_LINE
                print("progress:", int(100 * (i + 1) / len(mini_batches)), "%, ", "cost =", cost, end='\r')
        sys.stdout.write("\033[F")  # CURSOR_UP_ONE
        sys.stdout.write("\033[K")  # ERASE_LINE

        print("Done, cost of epoch", epoch + 1, ":", cost, "                                             ")

        error01_train, _ = LeNet5.Forward_Propagation(train_images, train_labels, 'test')
        err_rate_list.append(error01_train / 60000)
        # error01_test, _ = LeNet5.Forward_Propagation(test_images, test_labels, 'test')
        # err_rate_list.append([error01_train / 60000, error01_test / 10000])
        print("0/1 error of training set:", error01_train, "/", len(train_labels))
        # print("0/1 error of testing set: ", error01_test, "/", len(test_labels))
        print("Time used: ", time.time() - ste, "sec")
        print(
            "----------------------------------- epoch{} end -------------------------------------\n".format(epoch + 1))

        # conserve the model
        # with open('model_data_' + str(epoch) + '.pkl', 'wb') as output:
        #     pickle.dump(LeNet5, output, pickle.HIGHEST_PROTOCOL)

    err_rate_list = np.array(err_rate_list).T

    # This shows the error rate of training and testing data after each epoch
    # x = np.arange(epochs)
    # plt.xlabel('epochs')
    # plt.ylabel('error rate')
    # plt.plot(x, err_rate_list[0])
    # plt.plot(x, err_rate_list[1])
    # plt.legend(['training data', 'testing data'], loc='upper right')
    # plt.show()


# return random-shuffled mini-batches
def random_mini_batches(image, label, mini_batch_size=256, one_batch=False):
    m = image.shape[0]  # number of training examples
    mini_batches = []

    # Shuffle (image, label)
    permutation = list(np.random.permutation(m))
    shuffled_image = image[permutation, :, :, :]
    shuffled_label = label[permutation]

    # extract only one batch
    if one_batch:
        mini_batch_image = shuffled_image[0: mini_batch_size, :, :, :]
        mini_batch_label = shuffled_label[0: mini_batch_size]
        return mini_batch_image, mini_batch_label

    # Partition (shuffled_image, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_image = shuffled_image[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_label = shuffled_label[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_image, mini_batch_label)
        mini_batches.append(mini_batch)
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_image = shuffled_image[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_label = shuffled_label[num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_image, mini_batch_label)
        mini_batches.append(mini_batch)

    return mini_batches
