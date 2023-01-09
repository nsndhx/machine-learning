# coding=utf-8
from data_processing import *
from train import *
from LeNet5 import *

if __name__ == '__main__':
    # get the data
    train_images, train_labels, test_images, test_labels = load_data()

    print("Got data...\n")

    # data processing, normalization&zero-padding
    print("Normalization and zero-padding...\n")
    train_images = normalize(zero_pad(train_images[:, :, :, np.newaxis], 2), 'LeNet5')
    test_images = normalize(zero_pad(test_images[:, :, :, np.newaxis], 2), 'LeNet5')
    print("The shape of training image with padding: ", train_images.shape)
    print("The shape of testing image with padding: ", test_images.shape)
    print("Finish data processing...\n")

    # train LeNet5
    LeNet5 = LeNet5()
    print("Start training...")
    start_time = time.time()
    train(LeNet5, train_images, train_labels)
    end_time = time.time()
    print("Finished training, the total training time is {}s \n".format(end_time - start_time))

    # read model
    # with open('model_data_13.pkl', 'rb') as input_:
    #     LeNet5 = pickle.load(input_)

    # evaluate on test dataset
    print("Start testing...")
    error01, class_pred = LeNet5.Forward_Propagation(test_images, test_labels, 'test')
    print("error rate:", error01 / len(class_pred))
    print("Finished testing, the accuracy is {} \n".format(1 - error01 / len(class_pred)))
