import numpy as np

from pooling_utils import *
from convolution_utils import *
from activation_utils import *
from RBF_init_weight import *

bitmap = rbf_init_weight()


class LeNet5(object):
    """
    C1 -> S2 -> C3 -> S4 -> C5 -> F6 -> Output

    Reference: https://www.cnblogs.com/fengff/p/10173071.html
    """

    def __init__(self):
        # Designate combination of kernels and feature maps of S2.
        C3_mapping = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 0], [5, 0, 1],
                      [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 0], [4, 5, 0, 1],
                      [5, 0, 1, 2], [0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5], [0, 1, 2, 3, 4, 5]]

        kernal_shape = {"C1": (5, 5, 1, 6),
                       "C3": (5, 5, 6, 16),
                       "C5": (5, 5, 16, 120),
                       "F6": (120, 84),
                       "OUTPUT": (84, 10)}
        
        hyper_parameters_convolution = {"stride": 1, "pad": 0}
        hyper_parameters_pooling = {"stride": 2, "f": 2}

        self.C1 = ConvolutionLayer(kernal_shape["C1"], hyper_parameters_convolution)
        self.a1 = Activation("LeNet5_squash")
        self.S2 = PoolingLayer(hyper_parameters_pooling, "average")

        self.C3 = ConvolutionLayer_maps(kernal_shape["C3"], hyper_parameters_convolution, C3_mapping)
        self.a2 = Activation("LeNet5_squash")
        self.S4 = PoolingLayer(hyper_parameters_pooling, "average")

        self.C5 = ConvolutionLayer(kernal_shape["C5"], hyper_parameters_convolution)
        self.a3 = Activation("LeNet5_squash")

        self.F6 = FCLayer(kernal_shape["F6"])
        self.a4 = Activation("LeNet5_squash")

        self.Output = RBFLayer(bitmap)

    def Forward_Propagation(self, input_image, input_label, mode):
        self.label = input_label
        self.C1_FP = self.C1.forward_propagation(input_image)
        self.a1_FP = self.a1.forward_propagation(self.C1_FP)
        self.S2_FP = self.S2.forward_propagation(self.a1_FP)

        self.C3_FP = self.C3.forward_propagation(self.S2_FP)
        self.a2_FP = self.a2.forward_propagation(self.C3_FP)
        self.S4_FP = self.S4.forward_propagation(self.a2_FP)

        self.C5_FP = self.C5.forward_propagation(self.S4_FP)
        self.a3_FP = self.a3.forward_propagation(self.C5_FP)

        self.flatten = self.a3_FP[:, 0, 0, :]
        self.F6_FP = self.F6.forward_propagation(self.flatten)
        self.a4_FP = self.a4.forward_propagation(self.F6_FP)

        # output sum of the loss over mini-batch when mode = 'train'
        # output tuple of (0/1 error, class_predict) when mode = 'test'
        out = self.Output.forward_propagation(self.a4_FP, input_label, mode)

        return out

    def Back_Propagation(self, momentum, weight_decay):
        dy_pred = self.Output.back_propagation()

        dy_pred = self.a4.back_propagation(dy_pred)
        F6_BP = self.F6.back_propagation(dy_pred, momentum, weight_decay)
        reverse_flatten = F6_BP[:, np.newaxis, np.newaxis, :]

        reverse_flatten = self.a3.back_propagation(reverse_flatten)
        C5_BP = self.C5.back_propagation(reverse_flatten, momentum, weight_decay)

        S4_BP = self.S4.back_propagation(C5_BP)
        S4_BP = self.a2.back_propagation(S4_BP)
        C3_BP = self.C3.back_propagation(S4_BP, momentum, weight_decay)

        S2_BP = self.S2.back_propagation(C3_BP)
        S2_BP = self.a1.back_propagation(S2_BP)
        C1_BP = self.C1.back_propagation(S2_BP, momentum, weight_decay)

    # Stochastic Diagonal Levenberg-Marquaedt method for determining the learning rate before the beginning of each ep
    def SDLM(self, mu, lr_global):
        d2y_pred = self.Output.SDLM()
        d2y_pred = self.a4.SDLM(d2y_pred)

        F6_SDLM = self.F6.SDLM(d2y_pred, mu, lr_global)
        reverse_flatten = F6_SDLM[:, np.newaxis, np.newaxis, :]

        reverse_flatten = self.a3.SDLM(reverse_flatten)
        C5_SDLM = self.C5.SDLM(reverse_flatten, mu, lr_global)

        S4_SDLM = self.S4.SDLM(C5_SDLM)
        S4_SDLM = self.a2.SDLM(S4_SDLM)
        C3_SDLM = self.C3.SDLM(S4_SDLM, mu, lr_global)

        S2_SDLM = self.S2.SDLM(C3_SDLM)
        S2_SDLM = self.a1.SDLM(S2_SDLM)
        C1_SDLM = self.C1.SDLM(S2_SDLM, mu, lr_global)

# C1 layer
class ConvolutionLayer(object):
    def __init__(self, kernel_shape, parameters, init_mode='Gaussian_dist'):
        """
        :param kernel_shape: (n_f, n_f, n_C_prev, n_C)
        :param parameters: {"stride": s, "pad": p}
        :param init_mode:
        """
        self.parameters = parameters
        self.weight, self.bias = initialize(kernel_shape, init_mode)
        self.v_w, self.v_b = np.zeros(kernel_shape), np.zeros((1, 1, 1, kernel_shape[-1]))

    def forward_propagation(self, input_map):
        output_map, self.cache = conv_forward(input_map, self.weight, self.bias, self.parameters)
        return output_map

    def back_propagation(self, dZ, momentum, weight_decay):
        dA_prev, dW, db = conv_backward(dZ, self.cache)
        self.weight, self.bias, self.v_w, self.v_b = \
            update(self.weight, self.bias, dW, db, self.v_w, self.v_b, self.lr, momentum, weight_decay)
        return dA_prev

    def SDLM(self, d2Z, mu, lr_global):
        d2A_prev, d2W = conv_SDLM(d2Z, self.cache)
        h = np.sum(d2W) / d2Z.shape[0]
        self.lr = lr_global / (mu + h)
        return d2A_prev


# C3: convolution layer with assigned combination between input maps and weight
class ConvolutionLayer_maps(object):
    def __init__(self, kernel_shape, hyper_parameters, mapping, init_mode='Gaussian_dist'):
        """
        kernel_shape: (n_f, n_f, n_C_prev, n_C)
        hyper_parameters = {"stride": s, "pad": p}
        """
        self.hyper_parameters = hyper_parameters
        self.mapping = mapping
        self.wb = []  # list of [weight, bias]
        self.v_wb = []  # list of [v_w, v_b]
        for i in range(len(self.mapping)):
            weight_shape = (kernel_shape[0], kernel_shape[1], len(self.mapping[i]), 1)
            w, b = initialize(weight_shape, init_mode)
            self.wb.append([w, b])
            self.v_wb.append([np.zeros(w.shape), np.zeros(b.shape)])

    def forward_propagation(self, input_map):
        self.inputmap_shape = input_map.shape  # (n_m,14,14,6)
        self.caches = []
        output_maps = []
        for i in range(len(self.mapping)):
            output_map, cache = conv_forward(input_map[:, :, :, self.mapping[i]], self.wb[i][0], self.wb[i][1],
                                             self.hyper_parameters)
            output_maps.append(output_map)
            self.caches.append(cache)
        output_maps = np.swapaxes(np.array(output_maps), 0, 4)[0]
        return output_maps

    def back_propagation(self, dZ, momentum, weight_decay):
        dA_prevs = np.zeros(self.inputmap_shape)
        for i in range(len(self.mapping)):
            dA_prev, dW, db = conv_backward(dZ[:, :, :, i:i + 1], self.caches[i])
            self.wb[i][0], self.wb[i][1], self.v_wb[i][0], self.v_wb[i][1] = \
                update(self.wb[i][0], self.wb[i][1], dW, db, self.v_wb[i][0], self.v_wb[i][1], self.lr, momentum, weight_decay)
            dA_prevs[:, :, :, self.mapping[i]] += dA_prev
        return dA_prevs

        # Stochastic Diagonal Levenberg-Marquaedt

    def SDLM(self, d2Z, mu, lr_global):
        h = 0
        d2A_prevs = np.zeros(self.inputmap_shape)
        for i in range(len(self.mapping)):
            d2A_prev, d2W = conv_SDLM(d2Z[:, :, :, i:i + 1], self.caches[i])
            d2A_prevs[:, :, :, self.mapping[i]] += d2A_prev
            h += np.sum(d2W)
        self.lr = lr_global / (mu + h / d2Z.shape[0])
        return d2A_prevs


class PoolingLayer(object):
    def __init__(self, hyper_parameters, mode):
        self.hyper_parameters = hyper_parameters
        self.mode = mode

    def forward_propagation(self, input_map):  # n,28,28,6 -> n,10,10,16
        A, self.cache = pool_forward(input_map, self.hyper_parameters, self.mode)
        return A

    def back_propagation(self, dA):
        dA_prev = pool_backward(dA, self.cache, self.mode)
        return dA_prev

    def SDLM(self, d2A):
        d2A_prev = pool_backward(d2A, self.cache, self.mode)
        return d2A_prev


class Subsampling(object):
    def __init__(self, n_kernel, hyper_parameters):
        self.hyper_parameters = hyper_parameters
        self.weight = np.random.normal(0, 0.1, (1, 1, 1, n_kernel))
        self.bias = np.random.normal(0, 0.1, (1, 1, 1, n_kernel))
        self.v_w = np.zeros(self.weight.shape)
        self.v_b = np.zeros(self.bias.shape)

    def foward_prop(self, input_map):  # n,28,28,6 / n,10,10,16
        A, self.cache = subsampling_forward(input_map, self.weight, self.bias, self.hyper_parameters)
        return A

    def back_prop(self, dA, momentum, weight_decay):
        dA_prev, dW, db = subsampling_backward(dA, A_, weight, b, self.cache)
        self.weight, self.bias, self.v_w, self.v_b = \
            update(self.weight, self.bias, dW, db, self.v_w, self.v_b, self.lr, momentum, weight_decay)
        return dA_prev

    # Stochastic Diagonal Levenberg-Marquaedt
    def SDLM(self, d2A, mu, lr_global):
        d2A_prev, d2W, _ = subsampling_backward(dA, A_, weight, b, self.cache)
        h = np.sum(d2W) / d2A.shape[0]
        self.lr = lr_global / (mu + h)
        return d2A_prev


class Activation(object):
    def __init__(self, mode):
        (act, d_act), actfName = activation_func()
        act_index = actfName.index(mode)
        self.act = act[act_index]
        self.d_act = d_act[act_index]

    def forward_propagation(self, input_image):
        self.input_image = input_image
        return self.act(input_image)

    def back_propagation(self, dZ):
        dA = np.multiply(dZ, self.d_act(self.input_image))
        return dA

    # Stochastic Diagonal Levenberg-Marquardt
    def SDLM(self, d2Z):  # d2_LeNet5_squash
        dA = np.multiply(d2Z, np.power(self.d_act(self.input_image), 2))
        return dA


class FCLayer(object):
    def __init__(self, weight_shape, init_mode='Gaussian_dist'):
        self.v_w, self.v_b = np.zeros(weight_shape), np.zeros((weight_shape[-1],))
        self.weight, self.bias = initialize(weight_shape, init_mode)

    def forward_propagation(self, input_array):
        self.input_array = input_array  # (n_m, 120)
        return np.matmul(self.input_array, self.weight)  # (n_m, 84)

    def back_propagation(self, dZ, momentum, weight_decay):
        dA = np.matmul(dZ, self.weight.T)  # (256, 84) * (84, 120) = (256, 120)  (n_m, 84) * (84, 120) = (n_m, 120)
        dW = np.matmul(self.input_array.T, dZ)  # (256, 120).T * (256, 84) = (256, 1, 120, 84)  (n_m, 120).T * (n_m, 84) = (120, 84)
        db = np.sum(dZ.T, axis=1)  # (84,)
        self.weight, self.bias, self.v_w, self.v_b = update(self.weight, self.bias, dW, db, self.v_w, self.v_b, self.lr, momentum, weight_decay)
        return dA

    # Stochastic Diagonal Levenberg-Marquaedt
    def SDLM(self, d2Z, mu, lr_global):
        d2A = np.matmul(d2Z, np.power(self.weight.T, 2))
        d2W = np.matmul(np.power(self.input_array.T, 2), d2Z)
        h = np.sum(d2W) / d2Z.shape[0]
        self.lr = lr_global / (mu + h)
        return d2A


# not even slightly work
class RBFLayer_trainable_weight(object):
    def __init__(self, weight_shape, init_weight=None, init_mode='Gaussian_dist'):
        self.weight_shape = weight_shape  # =(10, 84)
        self.v_w = np.zeros(weight_shape)

        if init_weight.shape == (10, 84):
            self.weight = init_weight
        else:
            self.weight, _ = initialize(weight_shape, init_mode)

    def forward_propagation(self, input_array, label, mode):
        """
        :param input_array: (n_m, 84)
        :param label: (n_m, )
        :param mode:
        :return:
        """
        if mode == 'train':
            self.input_array = input_array
            self.weight_label = self.weight[label, :]  # (n_m, 84) labeled version of weight
            loss = 0.5 * np.sum(np.power(input_array - self.weight_label, 2), axis=1, keepdims=True)  # (n_m, )
            return np.sum(np.squeeze(loss))

        if mode == 'test':
            subtract_weight = (
                    input_array[:, np.newaxis, :] - np.array([self.weight] * input_array.shape[0]))  # (n_m,10,84)
            rbf_class = np.sum(np.power(subtract_weight, 2), axis=2)  # (n_m, 10)
            class_pred = np.argmin(rbf_class, axis=1)  # (n_m,)
            error01 = np.sum(label != class_pred)
            return error01, class_pred

    def back_propagation(self, label, lr, momentum, weight_decay):
        # n_m = label.shape[0]

        # d_output = np.zeros((n_m, n_class))
        # d_output[range(n_m), label] = 1    # (n_m, 10)  one-hot version of gradient w.r.t. output

        dy_predict = -self.weight_label + self.input_array  # (n_m, 84)

        dW_target = -dy_predict  # (n_m, 84)

        dW = np.zeros(self.weight_shape)  # (10,84)

        for i in range(len(label)):
            dW[label[i], :] += dW_target[i, :]

        self.v_w = momentum * self.v_w - weight_decay * lr * self.weight - lr * dW
        self.weight += self.v_w

        return dy_predict


class RBFLayer(object):
    def __init__(self, weight):
        self.weight = weight  # (10, 84)

    def forward_propagation(self, input_array, label, mode):
        """
        :param input_array: (n_m, 84)
        :param label: (n_m, )
        :param mode:
        :return:
        """
        if mode == 'train':
            self.input_array = input_array
            self.weight_label = self.weight[label, :]  # (n_m, 84) labeled version of weight
            loss = 0.5 * np.sum(np.power(input_array - self.weight_label, 2), axis=1, keepdims=True)  # (n_m, )
            return np.sum(np.squeeze(loss))
        if mode == 'test':
            # (n_m,1,84) - n_m*[(10,84)] = (n_m,10,84)
            subtract_weight = (
                    input_array[:, np.newaxis, :] - np.array([self.weight] * input_array.shape[0]))  # (n_m,10,84)
            rbf_class = np.sum(np.power(subtract_weight, 2), axis=2)  # (n_m, 10)
            class_pred = np.argmin(rbf_class, axis=1)  # (n_m,)
            error01 = np.sum(label != class_pred)
            return error01, class_pred

    def back_propagation(self):
        dy_predict = -self.weight_label + self.input_array  # (n_m, 84)
        return dy_predict

    def SDLM(self):
        # d2y_predict
        return np.ones(self.input_array.shape)


# initialization of the weights & bias
def initialize(kernel_shape, mode='Fan-in'):
    bias_shape = (1, 1, 1, kernel_shape[-1]) if len(kernel_shape) == 4 else (kernel_shape[-1],)
    if mode == 'Gaussian_dist':
        mu, sigma = 0, 0.1  # mu: mean value, sigma: standard deviation
        weight = np.random.normal(mu, sigma, kernel_shape)
        bias = np.ones(bias_shape) * 0.01
    elif mode == 'Fan-in':  # original init. in the paper
        Fi = np.prod(kernel_shape) / kernel_shape[-1]
        weight = np.random.uniform(-2.4 / Fi, 2.4 / Fi, kernel_shape)
        bias = np.ones(bias_shape) * 0.01
    return weight, bias


# update for the weights
def update(weight, bias, dW, db, vw, vb, lr, momentum=0, weight_decay=0):
    vw_u = momentum * vw - weight_decay * lr * weight - lr * dW
    vb_u = momentum * vb - weight_decay * lr * bias - lr * db
    weight_u = weight + vw_u
    bias_u = bias + vb_u
    return weight_u, bias_u, vw_u, vb_u
