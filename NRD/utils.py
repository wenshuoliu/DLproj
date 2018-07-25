""" The regularization of the parent matrix """
import keras.backend as K

class Regularizer(object):
    """Regularizer base class.
    """

    def __call__(self, x):
        return 0.

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Mat_reg(Regularizer):
    """Regularizer for parent matrix.
    # Arguments
        mat: numpy array; the matrix indicating the parent of each code.
        lamb: Float; the penalty tuning parameter. 
    """

    def __init__(self, mat, lamb, norm=2):
        self.lamb = K.cast_to_floatx(lamb)
        self.pmat = K.constant(value=mat, dtype=K.floatx(), name='parent_mat')
        self.norm = norm

    def __call__(self, embed_mat):
        diff = K.dot(self.pmat, embed_mat) #difference between each embedding and its parent
        if self.norm==2:
            return self.lamb*K.sum(K.square(diff))
        else:
            return self.lamb*K.sum(K.abs(diff))
