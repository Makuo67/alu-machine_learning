import numpy as np
import tensorflow as tf


class NST:
    """-----"""
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """-----"""
        if type(style_image) is not np.ndarray or \
           len(style_image.shape) != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if type(content_image) is not np.ndarray or \
           len(content_image.shape) != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        style_h, style_w, style_c = style_image.shape
        content_h, content_w, content_c = content_image.shape
        if style_h <= 0 or style_w <= 0 or style_c != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if content_h <= 0 or content_w <= 0 or content_c != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if (type(alpha) is not float and type(alpha) is not int) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if (type(beta) is not float and type(beta) is not int) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """-----"""
        if type(image) is not np.ndarray or len(image.shape) != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        h, w, c = image.shape
        if h <= 0 or w <= 0 or c != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        resized = tf.image.resize(np.expand_dims(
            image, axis=0), size=(h_new, w_new), method='bicubic')
        rescaled = resized / 255.0
        rescaled = tf.clip_by_value(rescaled, 0.0, 1.0)
        return rescaled

    def load_model(self):
        """-----"""
        VGG19_model = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')
        VGG19_model.trainable = False

        style_outputs = [VGG19_model.get_layer(
            name).output for name in self.style_layers]
        content_output = VGG19_model.get_layer(self.content_layer).output

        model_outputs = style_outputs + [content_output]
        self.model = tf.keras.models.Model(VGG19_model.input, model_outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """-----"""
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")
        if len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        input_layer = tf.cast(input_layer, tf.float64)
        _, h, w, c = input_layer.shape
        features = tf.reshape(input_layer, (h * w, c))
        gram = tf.matmul(features, features, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)
        gram /= tf.cast(h * w, tf.float64)
        return gram

    def generate_features(self):
        """-----"""
        VGG19_model = tf.keras.applications.vgg19
        preprocess_style = VGG19_model.preprocess_input(
            self.style_image * 255.0)
        preprocess_content = VGG19_model.preprocess_input(
            self.content_image * 255.0)

        style_features = self.model(preprocess_style)[:-1]
        content_feature = self.model(preprocess_content)[-1]

        gram_style_features = [self.gram_matrix(
            feature) for feature in style_features]

        self.gram_style_features = gram_style_features
        self.content_feature = content_feature

    def layer_style_cost(self, style_output, gram_target):
        """-----   """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
           len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")
        one, h, w, c = style_output.shape
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
           len(gram_target.shape) != 3 or gram_target.shape != (1, c, c):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c))

        style_output = tf.cast(style_output, tf.float64)
        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """    -----   """
        if type(style_outputs) is not list or len(
                style_outputs) != len(self.style_layers):
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    len(self.style_layers)))

        weight = 1.0 / float(len(self.style_layers))
        style_cost = tf.add_n(
            [self.layer_style_cost(style_outputs[i], self.gram_style_features[i]) * weight
             for i in range(len(self.style_layers))])
        return style_cost
