#### Some addons for keras ----

from keras.preprocessing.image import *  
from keras.preprocessing.image import _count_valid_files_in_directory

class FrameIterator(Iterator):
    """Iterator capable of reading images from a directory on disk, and labels 
    from a pandas dataframe
    # Arguments
        directory: Path to the directory to read images from.
            This directory will be considered to contain images from all classes. 
        dataframe: the pandas dataframe that contains the file_names and labels
        file_names: the column name of the dataframe for the file names in the directory
        labels: the columns of the dataframe for the labels
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    """

    def __init__(self, directory, dataframe, file_names, labels, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None, save_to_dir=None, save_prefix='', 
                 save_format='png',
                 follow_links=False,
                 interpolation='nearest'):
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        '''
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        '''
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        '''
        if subset is not None:
            validation_split = self.image_data_generator._validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError('Invalid subset name: ', subset,
                                 '; expected "training" or "validation"')
        else:
            split = None
        self.subset = subset
        '''
        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff', 'npy'}

        # first, count the number of samples and classes
        self.samples = 0
        self.samples = _count_valid_files_in_directory(self.directory, white_list_formats, follow_links)

        print('Found %d images.' % (self.samples))

        filenames = [f for f in os.listdir(directory) if not os.path.isdir(os.path.join(directory, f))]
       
        ext = set()
        for f in filenames:
            for e in white_list_formats:
                if f.lower().endswith('.'+e):
                    ext.add(e)
        if len(ext)>1:
            print("Image files must have the same extension.")
        ext = ext.pop()
            
        sub_df = dataframe[[file_names]+labels]
        sub_df = sub_df.set_index(file_names, drop=True)
        sub_df.index = [i+'.'+ext for i in sub_df.index]
        filenames = sorted([fn for fn in filenames if fn in sub_df.index])
        self.filenames = np.array(filenames)
        self.labels = sub_df.loc[filenames,:]
        
        super(FrameIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e7),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        label_dim = self.labels.shape[1]
        label_names = self.labels.columns
        batch_y = dict()
        for l in label_names:
            batch_y[l] = self.labels.loc[self.filenames[index_array], l].values
            
        
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
    
    

class ImageFrameGenerator(ImageDataGenerator):
    """Extension of ImageDataGenerator to generate batches with labels from 
    a pandas DataFrame. 
    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width, if < 1, or pixels if >= 1.
        height_shift_range: fraction of total height, if < 1, or pixels if >= 1.
        shear_range: shear intensity (shear angle in degrees).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channel.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
            Points outside the boundaries of the input are filled according to the given mode:
                'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                'nearest':  aaaaaaaa|abcd|dddddddd
                'reflect':  abcddcba|abcd|dcbaabcd
                'wrap':  abcdabcd|abcd|abcdabcd
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided. This is
            applied after the `preprocessing_function` (if any provided)
            but before any other transformation.
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    """    
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None):
        super(ImageFrameGenerator, self).__init__(featurewise_center=featurewise_center,
                 samplewise_center=samplewise_center,
                 featurewise_std_normalization=featurewise_std_normalization,
                 samplewise_std_normalization=samplewise_std_normalization,
                 zca_whitening=zca_whitening,
                 zca_epsilon=zca_epsilon,
                 rotation_range=rotation_range,
                 width_shift_range=width_shift_range,
                 height_shift_range=height_shift_range,
                 shear_range=shear_range,
                 zoom_range=zoom_range,
                 channel_shift_range=channel_shift_range,
                 fill_mode=fill_mode,
                 cval=cval,
                 horizontal_flip=horizontal_flip,
                 vertical_flip=vertical_flip,
                 rescale=rescale,
                 preprocessing_function=preprocessing_function,
                 data_format=data_format)
        
            
    def flow_from_frame(self, directory, dataframe, file_names, labels,
                 target_size=(256, 256), color_mode='rgb',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None, save_to_dir=None, save_prefix='', 
                 save_format='png',
                 follow_links=False,
                 interpolation='nearest'):
        return FrameIterator(directory, dataframe, file_names, labels, self,
                 target_size=target_size, color_mode=color_mode,
                 batch_size=batch_size, shuffle=shuffle, seed=seed,
                 data_format=self.data_format, save_to_dir=save_to_dir, save_prefix=save_prefix, 
                 save_format=save_format,
                 follow_links=follow_links,
                 interpolation=interpolation)
            
    