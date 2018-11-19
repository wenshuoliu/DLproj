import numpy as np
import keras.backend as K
from keras.engine.topology import InputSpec, Layer
from keras import activations, initializers, regularizers, constraints
from keras import objectives
import tensorflow as tf

class SetSum(Layer):
    """The DeepSet operation: apply a dense layer on the last axis, and then do a masked sum on axis=1. 

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        The input should have the dimensions: (batch, timestep, feature). The last layer should pass a mask with 
        dimension (batch, timestep), which is the default output_mask from Embedding(mask_zero=True)
    # Output shape
        (batch, feature). Note that the timestep dimension is summed out. 
    """
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(SetSum, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, mask=None):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
            
        if mask is not None:
            mask = K.cast(mask, 'float32')
            if not len(mask.shape) == 2:
                print("Mask shape:", mask.shape)
            mask = K.repeat(mask, self.units)
            mask = K.permute_dimensions(mask, (0, 2, 1))
            output = output*mask
            
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 3
        assert input_shape[-1]
        return (input_shape[0], self.units)
    
    def compute_mask(self, input, input_mask=None):
        return None

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class MaskedSum(Layer):
    """This layer is supposed to follow a Embedding layer or a MaskedDense layer. It sums over the non-masked embeddings.
    The summation is done along the timestep axis. 
    # Input shape
        (batch, timestep, feature). The last layer should pass a mask with 
        dimension (batch, timestep), which is the default output_mask from Embedding(mask_zero=True)
    # Output shape
        (batch, feature). Note that the timestep dimension is summed out.
    """
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskedSum, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        assert len(inputs.shape) == 3
        if mask is not None:
            mask = K.cast(mask, 'float32')
            assert len(mask.shape) == 2
            mask = K.repeat(mask, inputs.shape[-1])
            mask = K.permute_dimensions(mask, (0, 2, 1))
            masked = mask*inputs
            return K.sum(masked, axis=1)
        else:
            return K.sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def compute_mask(self, input, input_mask=None):
        return None
    
class MaskedPooling(Layer):
    """This layer is supposed to follow a Embedding layer or a MaskedDense layer. It does a max-pooling over the non-masked embeddings.
    The pooling is done along the timestep axis. 
    # Input shape
        (batch, timestep, feature). The last layer should pass a mask with 
        dimension (batch, timestep), which is the default output_mask from Embedding(mask_zero=True)
    # Output shape
        (batch, feature). Note that the timestep dimension is maxed out.
    """
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskedPooling, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        assert len(inputs.shape) == 3
        if mask is not None:
            mask = K.cast(mask, 'float32')
            assert len(mask.shape) == 2
            mask = K.repeat(mask, inputs.shape[-1])
            mask = K.permute_dimensions(mask, (0, 2, 1))
            masked = mask*inputs
            return K.max(masked, axis=1)
        else:
            return K.max(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def compute_mask(self, input, input_mask=None):
        return None
    
class MaskedDense(Layer):
    """The Dense layer that only works on the last axis, and propagates the mask from last layer.  

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        The input should have the dimensions: (batch, timestep, feature). The last layer should pass a mask with 
        dimension (batch, timestep), which is the default output_mask from Embedding(mask_zero=True)
    # Output shape
        (batch, feature, units). This Dense layer works on the feature axis, and output units dimension.  
    """
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MaskedDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, mask=None):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)   
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)
    
    def compute_mask(self, input, input_mask=None):
        return input_mask

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
##complete one


def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)


class SkipgramRegularization(Layer):
    """Layer written by Wenyi to do dynamic training with skipgram. 
    """
    def __init__(self, num_samples, num_classes, lamb, **kwargs):
        self.num_samples = num_samples #number of netagive sample
        self.num_classes = num_classes #number of code categories
        self.lamb = lamb #tuning parameter to control the weight between skipgram and prediction model
        super(SkipgramRegularization, self).__init__(**kwargs)
        
    def build(self, input_shape):
        dense_shape, classes_shape = input_shape
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.num_classes, dense_shape[2]),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.num_classes,),
                                    initializer='uniform',
                                    trainable=True) 
        super(SkipgramRegularization, self).build(input_shape)

    def call(self,x):
        inputs_in, labels_in = x
        inputs_in = tf.cast(inputs_in, tf.float32)
        labels_in = tf.cast(labels_in, tf.float32)
        total_loss = 0
        n_codes = labels_in.shape[1]
        for i in range(n_codes):
            inputs = tf.gather(inputs_in, i, axis=1)
            for j in range(i+1, n_codes):
                labels = tf.gather(labels_in, [j], axis=1)
                loss = tf.nn.sampled_softmax_loss(
                        weights = self.kernel,
                        biases = self.bias,
                        labels = labels,
                        inputs = inputs,
                        num_sampled=self.num_samples,
                        num_classes=self.num_classes,
                        num_true=1)
                labels_i = tf.gather(labels_in, [i], axis=1)
                condition = tf.logical_or(tf.equal(labels, np.float32(0.)), tf.equal(labels_i, np.float32(0.)))
                total_loss += loss*tf.to_float(tf.logical_not(condition))
                
        cost = self.lamb*tf.reduce_mean(loss)
        self.add_loss(cost,x)
        #you can output whatever you need, just update output_shape adequately
        #But this is probably useful
        return cost

    def compute_output_shape(self, input_shape):
        dense_shape,classes_shape = input_shape
        return (dense_shape[0],)