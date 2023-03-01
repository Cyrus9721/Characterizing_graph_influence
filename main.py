# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com
# Modified by Zizhang Chen

import numpy as np
from generate_mid_layer_feature import FeatureExtraction
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from model_softmax import SimplifiedGraphNeuralNetwork
from sklearn.metrics import log_loss
from scipy.special import softmax, log_softmax
from matplotlib import pyplot as plt


# from model import LogisticRegression


def main():
    """ dataset """

    iris = load_iris()
    x, y = iris.data, iris.target
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=123)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(train_y.reshape(-1, 1))

    one_hot_labels_train = enc.transform(train_y.reshape(-1, 1)).toarray()
    one_hot_labels_test = enc.transform(test_y.reshape(-1, 1)).toarray()

    """ Train Logistic Regression """
    lr = SimplifiedGraphNeuralNetwork(l2_reg=1.0, fit_intercept=True)
    lr.fit(train_x, train_y, sample_weight=None, verbose=False)
    logits_test_y = test_x @ lr.model.coef_.T + lr.model.intercept_
    logits_train_y = train_x @ lr.model.coef_.T + lr.model.intercept_

    ori_val_loss, ave_ori_val_loss = lr.log_loss(logits_test_y, one_hot_labels_test)

    numpy_theoritic_loss = log_loss(test_y, softmax(logits_test_y, axis=1))

    # Check whether the log loss is correct:

    print('over all log loss', ori_val_loss)
    print('sklearn implemented log loss', numpy_theoritic_loss)
    print('our implemented log loss', ave_ori_val_loss)

    train_total_grad, train_indiv_grad = lr.grad(train_x, logits_train_y, one_hot_labels_train)
    val_loss_total_grad, val_loss_indiv_grad = lr.grad(test_x, logits_test_y, one_hot_labels_test)

    hessian_no_reg, hess, hessian_reg_term = lr.hess(train_x, logits_train_y)

    loss_grad_hvp = lr.get_inv_hvp(hess, val_loss_total_grad.T)

    pred_infl = train_indiv_grad.dot(loss_grad_hvp)

    pred_infl = list(pred_infl.reshape(-1))
    #
    num_train = len(train_x)
    act_infl = []
    print(pred_infl)

    for i in range(num_train):
        sample_weight = np.ones(num_train)
        sample_weight[i] = 0

        # lr.fit(train_x, train_y, sample_weight=sample_weight)
        # new_val_loss, _ = lr.log_loss(logits_test_y, one_hot_labels_test)
        # act_infl.append(new_val_loss - ori_val_loss)
        lr_new = SimplifiedGraphNeuralNetwork(l2_reg=1.0, fit_intercept=True)
        lr_new.fit(train_x, train_y, )

        # act_infl.append(new_val_loss - ori_val_loss)

    # print(pred_infl)
    print(act_infl)
    # plt.plot(act_infl, pred_infl)
    # plt.show()

    return


if __name__ == "__main__":
    main()
