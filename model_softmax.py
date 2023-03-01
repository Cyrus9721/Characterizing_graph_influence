from abc import ABC

import numpy as np
import pandas as pd
import random
import tensorflow.compat.v1 as tf
import cupy as cp
from graph_neural_networks import SGC_layer2
from model import IFBaseClass
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax, log_softmax
from tqdm import tqdm
from sklearnex import patch_sklearn, config_context

patch_sklearn()


class SimplifiedGraphNeuralNetwork(IFBaseClass):
    """
    Simplified Graph Neural Network with 1 layer, this is equivalent to fit a logistic regression with softmax loss
    borrowed from https://github.com/kohpangwei/group-influence-release/blob/master/influence/logistic_regression.py
    """

    def __init__(self, l2_reg=0, fit_intercept=False, warm_start=False, random_state=0, tol=1e-4, solver='lbfgs',
                 max_iter=2048):
        # super.__init__(SimplifiedGraphNeuralNetwork)
        self.weight = None
        self.l2_reg = l2_reg
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.random_state = random_state
        self.model = LogisticRegression(
            penalty="l2",
            C=(1.0 / self.l2_reg),
            fit_intercept=self.fit_intercept,
            tol=tol,
            solver=solver,
            max_iter=max_iter,
            multi_class='multinomial',
            warm_start=self.warm_start,
            random_state=self.random_state
        )

    def log_loss(self, logits, one_hot_labels, sample_weight=None, l2_reg=False, eps=1e-15):
        """
        L_{\log}(y, y_hat) = -y * log (y_hat)
        L_{reg} = - l2_reg * |w|_{2} / 2
        :param logits: array-like of float, raw output of simplified graph neural network, shape = (n, number of classes).
        :param one_hot_labels: array-like of float, one hot labels of ground truth, shape = (n, number of classes).
        :param sample_weight: array-like of float, sample weights for each feature
        :param l2_reg: l2_regularization term
        :param eps: float
        :return: float, loss, average loss
        """
        n = len(logits)

        sample_weight = self.set_sample_weight(n, sample_weight)

        softmax_pred = softmax(logits, axis=1)

        log_softmax_pred = np.log(softmax_pred + eps)

        indiv_cross_entropy = -np.sum(np.multiply(one_hot_labels, log_softmax_pred), axis=1)

        indiv_cross_entropy = np.multiply(sample_weight, indiv_cross_entropy)

        cross_entropy = np.sum(indiv_cross_entropy)

        average_cross_entropy = cross_entropy / n

        if l2_reg:
            reg_loss = self.l2_reg * np.linalg.norm(self.model.coef_, ord=2) / 2
            # reg_loss = self.l2_reg * np.linalg.norm(self.model.coef_, ord=2)
            cross_entropy += reg_loss
            average_cross_entropy += reg_loss

        """
        The original way of calculating the log softmax loss with regularization is confusing, 
        """
        # indiv_loss = cross_entropy
        #
        # total_loss_no_reg = tf.reduce_sum(tf.multiply(cross_entropy, sample_weight),
        #                                       name='total_loss_no_reg')
        #
        # tf.add_to_collection('losses', indiv_loss)
        #
        # total_loss_reg = tf.add_n(tf.get_collection('losses'), name='total_loss_reg')
        # avg_loss_reg = total_loss_reg / tf.cast(tf.shape(logits)[0], tf.float64)

        return cross_entropy, average_cross_entropy

    def grad(self, x, logits, one_hot_labels, sample_weight=None, l2_reg=None, fit_intercept=True):
        """
        Explicitly computes the softmax gradients by thetha.
        grad_theta_i loss(x, y) = -([i == y] - softmax_i) * x
        grad_b_i loss(x, y) = -([i == y] - softmax_i)
        :param x: array-like of float feature input
        :param logits: array-like of float raw output of SGC
        :param one_hot_labels: array-like of float one hot labels ( ground truth )
        :param sample_weight: sample weights
        :param l2_reg: default set to regular term
        :param fit_intercept:
        :return:
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        n = len(logits)

        K = one_hot_labels.shape[1]  # num_classes

        D = x.shape[1]  # num_dimensions

        sample_weight = self.set_sample_weight(n, sample_weight)
        sample_weight = sample_weight.reshape(-1, 1)

        softmax_pred = softmax(logits, axis=1)

        factor = -(one_hot_labels - softmax_pred)
        assert (factor.ndim == 2)
        expand_factor = np.expand_dims(factor, axis=2)  # (n, num_classes, 1)
        expand_x = np.expand_dims(x, 1)  # (n, 1, num_dimension)

        indiv_grad = np.multiply(expand_factor, expand_x)  # (n, num_classes, num_dimension)
        indiv_grad = indiv_grad.reshape(-1, K * D)  # (n, num_classes * num_dimension)

        weighted_indiv_grad = indiv_grad * sample_weight

        grad_reg = self.l2_reg * self.model.coef_.reshape(-1, K * D)

        if fit_intercept:
            weighted_indiv_grad = np.concatenate([weighted_indiv_grad, factor], axis=1)
            grad_reg = np.concatenate([grad_reg, np.zeros(K).reshape(1, -1)], axis=1)

        total_grad = np.sum(weighted_indiv_grad, axis=0).reshape(1, -1)

        if l2_reg:
            total_grad += grad_reg

        return total_grad, weighted_indiv_grad

    def grad_sum(self, x, logits, one_hot_labels, sample_weight=None, l2_reg=None, fit_intercept=True):
        """
        Explicitly computes the softmax gradients by thetha.
        grad_theta_i loss(x, y) = -([i == y] - softmax_i) * x
        grad_b_i loss(x, y) = -([i == y] - softmax_i)
        :param x: array-like of float feature input
        :param logits: array-like of float raw output of SGC
        :param one_hot_labels: array-like of float one hot labels ( ground truth )
        :param sample_weight: sample weights
        :param l2_reg: default set to regular term
        :param fit_intercept:
        :return:
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        n = len(logits)

        K = one_hot_labels.shape[1]  # num_classes

        D = x.shape[1]  # num_dimensions

        sample_weight = self.set_sample_weight(n, sample_weight)
        sample_weight = sample_weight.reshape(-1, 1)

        softmax_pred = softmax(logits, axis=1)

        factor = -(one_hot_labels - softmax_pred)
        assert (factor.ndim == 2)

        # factor = np.sum(factor, axis=0).reshape(1, factor.shape[1])
        # x = np.sum(x, axis=0).reshape(1, x.shape[1])


        expand_factor = np.expand_dims(factor, axis=2)  # (n, num_classes, 1)
        expand_x = np.expand_dims(x, 1)  # (n, 1, num_dimension)

        sum_grad = 0.0
        # for i in tqdm(range(n)):
        # print(n)
        for i in range(n):
            # print(expand_factor[i])
            # print(expand_x[i])
            sum_grad += np.multiply(expand_factor[i], expand_x[i])
        sum_grad = np.array(sum_grad)

        # print(sum_grad.shape)

        sum_grad = sum_grad.reshape(-1, K * D)

        # indiv_grad = np.multiply(expand_factor, expand_x)  # (1, num_classes, num_dimension)
        # indiv_grad = indiv_grad.reshape(-1, K * D)  # (1, num_classes * num_dimension)

        # weighted_indiv_grad = indiv_grad * sample_weight

        # grad_reg = self.l2_reg * self.model.coef_.reshape(-1, K * D)
        # print(weighted_indiv_grad.shape)
        sum_factor = np.sum(factor, axis=0).reshape(1, factor.shape[1])
        if fit_intercept:
            weighted_indiv_grad = np.concatenate([sum_grad, sum_factor], axis=1)
            # grad_reg = np.concatenate([grad_reg, np.zeros(K).reshape(1, -1)], axis=1)

        # total_grad = np.sum(weighted_indiv_grad, axis=0).reshape(1, -1)

        # if l2_reg:
        #     total_grad += grad_reg

        return weighted_indiv_grad

    def grad_one_batch(self, x, logits, one_hot_labels, sample_weight=None, l2_reg=None, fit_intercept=True,
                       batch_index=None):
        """
        Explicitly computes the softmax gradients by thetha.
        grad_theta_i loss(x, y) = -([i == y] - softmax_i) * x
        grad_b_i loss(x, y) = -([i == y] - softmax_i)
        :param x: array-like of float feature input
        :param logits: array-like of float raw output of SGC
        :param one_hot_labels: array-like of float one hot labels ( ground truth )
        :param sample_weight: sample weights
        :param l2_reg: default set to regular term
        :param fit_intercept:
        :return:
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # n = len(logits)
        n = len(batch_index)

        K = one_hot_labels.shape[1]  # num_classes

        D = x.shape[1]  # num_dimensions

        sample_weight = self.set_sample_weight(n, sample_weight)
        sample_weight = sample_weight.reshape(-1, 1)

        x = x[batch_index]
        logits = logits[batch_index]
        one_hot_labels = one_hot_labels[batch_index]

        softmax_pred = softmax(logits, axis=1)

        factor = -(one_hot_labels - softmax_pred)
        assert (factor.ndim == 2)
        expand_factor = np.expand_dims(factor, axis=2)  # (n, num_classes, 1)
        expand_x = np.expand_dims(x, 1)  # (n, 1, num_dimension)

        indiv_grad = np.multiply(expand_factor, expand_x)  # (n, num_classes, num_dimension)
        indiv_grad = indiv_grad.reshape(-1, K * D)  # (n, num_classes * num_dimension)

        weighted_indiv_grad = indiv_grad * sample_weight

        grad_reg = self.l2_reg * self.model.coef_.reshape(-1, K * D)

        if fit_intercept:
            weighted_indiv_grad = np.concatenate([weighted_indiv_grad, factor], axis=1)
            grad_reg = np.concatenate([grad_reg, np.zeros(K).reshape(1, -1)], axis=1)

        total_grad = np.sum(weighted_indiv_grad, axis=0).reshape(1, -1)

        if l2_reg:
            total_grad += grad_reg

        return total_grad, weighted_indiv_grad

    def grad_x(self, x, logits, one_hot_labels, sample_weight=None, l2_reg=None, fit_intercept=True):
        """
        This function explicitly calculate the gradient of predicted x
        :param x:
        :param logits:
        :param one_hot_labels:
        :param sample_weight:
        :param l2_reg:
        :param fit_intercept:
        :return:
        """

    def hess(self, x, logits, sample_weight=None, l2_reg=None, fit_intercept=True):
        """
        Explicitly computes the softmax hessian.
        grad_theta_i grad_theta_j loss(x, y)
            = softmax_i ([i == j] - softmax_j) x x^T
        grad_theta_i grad_b_j loss(x, y)
            = softmax_i ([i == j] - softmax_j) x
        grad_b_i grad_b_j loss(x, y)
            = softmax_i ([i == j] - softmax_j)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        n = len(logits)

        K = logits.shape[1]  # num_classes

        D = x.shape[1]  # num_dimensions

        KD = K * D
        sample_weight = self.set_sample_weight(n, sample_weight)

        softmax_pred = softmax(logits, axis=1)

        factor = tf.linalg.diag(softmax_pred) - \
                 tf.einsum('ai,aj->aij', softmax_pred, softmax_pred)  # (?, Kp, Kp)
        indiv_hessian = tf.reshape(
            tf.einsum('aij,ak,al->aikjl', factor, x, x),  # (?, Kp, D, Kp, D)
            (-1, KD, KD))  # (?, KpD, KpD)

        # Hessian of l2 regularization
        hess_reg = self.l2_reg * tf.eye(KD, KD)

        if fit_intercept:
            off_diag = tf.reshape(
                tf.einsum('aij,ak->aijk', factor, x),  # (?, Kp, Kp, D)
                (-1, K, KD))  # (?, Kp, KpD)

            top_row = tf.concat([indiv_hessian,
                                 tf.transpose(off_diag, (0, 2, 1))], axis=2)
            bottom_row = tf.concat([off_diag, factor], axis=2)
            indiv_hessian = tf.concat([top_row, bottom_row], axis=1)

            hess_reg = tf.pad(hess_reg, [[0, K], [0, K]],
                              mode="CONSTANT", constant_values=0.0).numpy()

        hessian_no_reg = tf.einsum('aij,a->ij', indiv_hessian, sample_weight).numpy()

        hessian_reg = hessian_no_reg + hess_reg
        hessian_reg_term = hess_reg

        return hessian_no_reg, hessian_reg, hessian_reg_term

    def hess_cuda(self, x, logits, sample_weight=None, l2_reg=None, fit_intercept=True):

        n = len(logits)

        K = logits.shape[1]

        D = x.shape[1]

        KD = K * D
        softmax_pred = softmax(logits, axis=1)

        temp_pred = softmax_pred[0]

        temp_x = x[0]

        temp_factor = cp.diag(temp_pred) - cp.einsum('i, j -> ij', temp_pred, temp_pred)

        temp_indiv_hessian = cp.einsum('ij, k, l -> ikjl', temp_factor, temp_x, temp_x).reshape(KD, KD)

        temp_off_diag = cp.einsum('ij, k -> ijk', temp_factor, temp_x).reshape(K, KD)

        temp_top_row = cp.concatenate([temp_indiv_hessian, temp_off_diag.T], axis=1)

        temp_bottom_row = cp.concatenate([temp_off_diag, temp_factor], axis=1)

        temp_indiv_hessian_all = cp.concatenate([temp_top_row, temp_bottom_row])

        del temp_off_diag
        del temp_top_row
        del temp_bottom_row
        del temp_indiv_hessian
        del temp_pred
        del temp_x
        del temp_factor

        for i in tqdm(range(1, n)):
            temp_pred = softmax_pred[i]

            temp_x = x[i]

            temp_factor = cp.diag(temp_pred) - cp.einsum('i, j -> ij', temp_pred, temp_pred)

            temp_indiv_hessian = cp.einsum('ij, k, l -> ikjl', temp_factor, temp_x, temp_x).reshape(KD, KD)

            # with intercept
            temp_off_diag = cp.einsum('ij, k -> ijk', temp_factor, temp_x).reshape(K, KD)

            temp_top_row = cp.concatenate([temp_indiv_hessian, temp_off_diag.T], axis=1)

            temp_bottom_row = cp.concatenate([temp_off_diag, temp_factor], axis=1)

            temp_indiv_hessian = cp.concatenate([temp_top_row, temp_bottom_row])

            temp_indiv_hessian_all += temp_indiv_hessian

            del temp_factor
            del temp_indiv_hessian
            del temp_off_diag
            del temp_top_row
            del temp_bottom_row
            del temp_pred
            del temp_x

        temp_indiv_hessian_all += cp.pad(cp.eye(KD, KD) * self.l2_reg,
                                         [[0, K], [0, K]], mode='constant', constant_values=0)

        return temp_indiv_hessian_all

    def fit(self, x, y, sample_weight=None, verbose=False):
        sample_weight = self.set_sample_weight(x.shape[0], sample_weight)

        self.model.fit(x, y, sample_weight=sample_weight)
        self.weight: np.ndarray = self.model.coef_.flatten()
        if self.fit_intercept:
            self.bias: np.ndarray = self.model.intercept_

        if verbose:
            pred, _ = self.pred(x)
            train_loss_wo_reg = self.log_loss(x, y, sample_weight)
            reg_loss = np.sum(np.power(self.weight, 2)) * self.l2_reg / 2.
            train_loss_w_reg = train_loss_wo_reg + reg_loss

            print("Train loss: %.5f + %.5f = %.5f" % (train_loss_wo_reg, reg_loss, train_loss_w_reg))

        return

    def pred(self, x):
        return self.model.predict_proba(x)[:, 1], self.model.predict(x)


def fast_hess(x, logits):
    n = len(logits)

    K = logits.shape[1]

    D = x.shape[1]

    KD = K * D

    softmax_pred = softmax(logits, axis=1)

    temp_pred = softmax_pred[0]
    temp_x = x[0]

    temp_factor = np.diag(temp_pred) - np.einsum('i, j -> ij', temp_pred, temp_pred)

    temp_indiv_hessian = np.einsum('ij, k, l -> ikjl', temp_factor, temp_x, temp_x).reshape(KD, KD)

    temp_off_diag = np.einsum('ij, k -> ijk', temp_factor, temp_x).reshape(K, KD)

    temp_top_row = np.concatenate([temp_indiv_hessian, temp_off_diag.T], axis=1)

    temp_bottom_row = np.concatenate([temp_off_diag, temp_factor], axis=1)

    temp_indiv_hessian_all = np.concatenate([temp_top_row, temp_bottom_row])

    del temp_off_diag
    del temp_top_row
    del temp_bottom_row
    del temp_pred
    del temp_x
    del temp_factor

    for i in tqdm(range(1, n)):
        temp_pred = softmax_pred[i]

        temp_x = x[i]

        temp_factor = np.diag(temp_pred) - np.einsum('i, j -> ij', temp_pred, temp_pred)

        temp_indiv_hessian = np.einsum('ij, k, l -> ikjl', temp_factor, temp_x, temp_x).reshape(KD, KD)

        # with intercept
        temp_off_diag = np.einsum('ij, k -> ijk', temp_factor, temp_x).reshape(K, KD)

        temp_top_row = np.concatenate([temp_indiv_hessian, temp_off_diag.T], axis=1)

        temp_bottom_row = np.concatenate([temp_off_diag, temp_factor], axis=1)

        temp_indiv_hessian = np.concatenate([temp_top_row, temp_bottom_row])

        temp_indiv_hessian_all += temp_indiv_hessian

    temp_indiv_hessian_all += np.pad(np.eye(KD, KD) * 1.0,
                                     [[0, K], [0, K]], mode='constant', constant_values=0)
    return temp_indiv_hessian_all


def fast_hess_cuda(x, logits):
    n = len(logits)

    K = logits.shape[1]

    D = x.shape[1]

    KD = K * D

    softmax_pred = softmax(logits, axis=1)

    temp_pred = softmax_pred[0]
    temp_x = x[0]

    temp_factor = cp.diag(temp_pred) - cp.einsum('i, j -> ij', temp_pred, temp_pred)

    temp_indiv_hessian = cp.einsum('ij, k, l -> ikjl', temp_factor, temp_x, temp_x).reshape(KD, KD)

    temp_off_diag = cp.einsum('ij, k -> ijk', temp_factor, temp_x).reshape(K, KD)

    temp_top_row = cp.concatenate([temp_indiv_hessian, temp_off_diag.T], axis=1)

    temp_bottom_row = cp.concatenate([temp_off_diag, temp_factor], axis=1)

    temp_indiv_hessian_all = cp.concatenate([temp_top_row, temp_bottom_row])

    del temp_off_diag
    del temp_top_row
    del temp_bottom_row
    del temp_indiv_hessian
    del temp_pred
    del temp_x
    del temp_factor

    for i in tqdm(range(1, n)):
        temp_pred = softmax_pred[i]

        temp_x = x[i]

        temp_factor = cp.diag(temp_pred) - cp.einsum('i, j -> ij', temp_pred, temp_pred)

        temp_indiv_hessian = cp.einsum('ij, k, l -> ikjl', temp_factor, temp_x, temp_x).reshape(KD, KD)

        # with intercept
        temp_off_diag = cp.einsum('ij, k -> ijk', temp_factor, temp_x).reshape(K, KD)

        temp_top_row = cp.concatenate([temp_indiv_hessian, temp_off_diag.T], axis=1)

        temp_bottom_row = cp.concatenate([temp_off_diag, temp_factor], axis=1)

        temp_indiv_hessian = cp.concatenate([temp_top_row, temp_bottom_row])

        temp_indiv_hessian_all += temp_indiv_hessian

        del temp_factor
        del temp_indiv_hessian
        del temp_off_diag
        del temp_top_row
        del temp_bottom_row
        del temp_pred
        del temp_x

    temp_indiv_hessian_all += cp.pad(cp.eye(KD, KD) * 1.0,
                                     [[0, K], [0, K]], mode='constant', constant_values=0)

    return temp_indiv_hessian_all


def fast_get_inv_hvp_cuda(hessian, vectors, cholskey=False):
    vectors_cuda = cp.array(vectors)
    if cholskey:
        return cp.linalg.solve(hessian, vectors_cuda)
    else:
        return cp.linalg.pinv(hessian).dot(vectors_cuda.T)
        # return cp.linalg.pinv(hessian).dot(vectors_cuda)
