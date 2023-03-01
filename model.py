# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

from abc import ABC, abstractmethod
from typing import Sequence, Tuple
import numpy as np
from scipy.linalg import cho_solve, cho_factor


class IFBaseClass(ABC):
    """ This class is modified for influence function on Simplified Graph Neural Network"""

    @staticmethod
    def set_sample_weight(n: int, sample_weight: np.ndarray or Sequence[float] = None) -> np.ndarray:
        if sample_weight is None:
            sample_weight = np.ones(n)
        else:
            if isinstance(sample_weight, np.ndarray):
                assert sample_weight.shape[0] == n
            elif isinstance(sample_weight, (list, tuple)):
                assert len(sample_weight) == n
                sample_weight = np.array(sample_weight)
            else:
                raise TypeError

            assert min(sample_weight) >= 0
            assert max(sample_weight) <= 2

        return sample_weight

    @staticmethod
    def set_feature_weight(n: int, feature_weight: np.ndarray or Sequence[float] = None) -> np.ndarray:
        """
        For two layer graph neural network, we first want to investigate the effect of change of layers on the loss
        """
        if feature_weight is None:
            feature_weight = np.ones(n)
        else:
            if isinstance(feature_weight, np.ndarray):
                assert feature_weight.shape[0] == n
            elif isinstance(feature_weight, (list, tuple)):
                assert len(feature_weight) == n
                feature_weight = np.array(feature_weight)
            else:
                raise TypeError

        assert min(feature_weight) >= 0
        assert max(feature_weight) <= 2

        return feature_weight

    @staticmethod
    def check_pos_def(M: np.ndarray) -> bool:
        pos_def = np.all(np.linalg.eigvals(M) > 0)
        print("Hessian positive definite: %s" % pos_def)
        return pos_def

    @staticmethod
    def get_inv_hvp(hessian: np.ndarray, vectors: np.ndarray, cho: bool = True, use_eps: bool = False,
                    eps: float = 1e-15, pseudo: bool = False) -> np.ndarray:
        if pseudo:
            hess_inv = np.linalg.pinv(hessian)
            return hess_inv.dot(vectors.T)
        if cho:
            return cho_solve(cho_factor(hessian), vectors)
        elif use_eps:
            hess_inv = np.linalg.inv(hessian)
            return hess_inv.dot(vectors.T)
        else:
            a = np.zeros(hessian.shape).astype(float)
            np.fill_diagonal(a, eps)
            return cho_solve(cho_factor(hessian + a), vectors)

    @abstractmethod
    def log_loss(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray or Sequence[float] = None,
                 l2_reg: bool = False) -> float:
        raise NotImplementedError

    @abstractmethod
    def grad(self, x: np.ndarray, y: np.ndarray, onehot: np.ndarray,
             sample_weight: np.ndarray or Sequence[float] = None,
             l2_reg: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """ Return the sum of all gradients and every individual gradient """
        raise NotImplementedError

    # @abstractmethod
    def grad_pred(self, x: np.ndarray, sample_weight: np.ndarray or Sequence[float] = None) -> Tuple[
        np.ndarray, np.ndarray]:
        """ Return the sum of all gradients and every individual gradient """
        raise NotImplementedError

    @abstractmethod
    def hess(self, x: np.ndarray, logits: np.ndarray, sample_weight: np.ndarray or Sequence[float] = None,
             check_pos_def: bool = False) -> np.ndarray:
        raise NotImplementedError

    # @abstractmethod
    def hess_cuda(self, x: np.ndarray, logits: np.ndarray, sample_weight: np.ndarray or Sequence[float] = None,
                  check_pos_def: bool = False) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def pred(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Return the predictive probability and class label """
        raise NotImplementedError
        # pass

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray or Sequence[float] = None) -> None:
        raise NotImplementedError
