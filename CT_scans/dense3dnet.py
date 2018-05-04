from keras import backend as K
from keras.models import Model
from keras.layers import GlobalAveragePooling3D, Input, ZeroPadding3D, BatchNormalization, MaxPooling3D, Concatenate, AveragePooling3D
from keras.layers.core import Dense, Activation, Dropout, Lambda
from keras.layers.convolutional import Conv3D

def conv_block(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv3D(4 * growth_rate, 1, use_bias=False,
                name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv3D(growth_rate, 3, padding='same', use_bias=False,
                name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def dense_block(x, n_blocks, growth_rate, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(n_blocks):
        x = conv_block(x, growth_rate, name=name + '_block' + str(i + 1))
    return x

def transition_block(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv3D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False,
               name=name + '_conv')(x)
    x = AveragePooling3D(2, strides=2, name=name + '_pool')(x)
    return x

def Dense3DNet(blocks,
               growth_rate=12,
               reduction=0.5,
             input_tensor=None,
             input_shape=(256, 256, 192, 1),
             pooling=None):
    """Instantiates the DenseNet architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with
    TensorFlow, Theano, and CNTK. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        blocks: numbers of building blocks for the four dense layers.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 4 if K.image_data_format() == 'channels_last' else 1

    x = ZeroPadding3D(padding=3)(img_input)
    x = Conv3D(2*growth_rate, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding3D(padding=1)(x)
    x = MaxPooling3D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], growth_rate, name='conv2')
    x = transition_block(x, reduction, name='pool2')
    x = dense_block(x, blocks[1], growth_rate, name='conv3')
    x = transition_block(x, reduction, name='pool3')
    x = dense_block(x, blocks[2], growth_rate, name='conv4')
    x = transition_block(x, reduction, name='pool4')
    x = dense_block(x, blocks[3], growth_rate, name='conv5')

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling3D(name='avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling3D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    model = Model(inputs=inputs, outputs=x)
    return model