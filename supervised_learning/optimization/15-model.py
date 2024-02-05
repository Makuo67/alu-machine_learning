#!/usr/bin/env python3
"""All together"""
import tensorflow as tf
import numpy as np


def model(Data_train,
          Data_valid,
          layers,
          activations,
          alpha=0.001,
          beta1=0.9,
          beta2=0.999,
          epsilon=1e-8,
          decay_rate=1,
          batch_size=32,
          epochs=5,
          save_path='/tmp/model.ckpt'):
    """Separate training and validation data"""
    with tf.Session() as sess:
        sess.run(init)
    
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    m, input_size = X_train.shape
    _, output_size = Y_train.shape

    # Build the neural network model
    x = tf.placeholder(tf.float32, shape=(None, input_size), name='x')
    y = tf.placeholder(tf.float32, shape=(None, output_size), name='y')
    
    # Initialize weights and biases
    parameters = initialize_parameters(layers)

    # Forward propagation
    y_pred = forward_propagation(x, parameters, activations)

    # Compute cost
    cost = compute_cost(y_pred, y)

    # Define the optimizer
    optimizer = create_optimizer(
        alpha, beta1, beta2, epsilon, decay_rate)

    # Training operation
    train_op = optimizer.minimize(cost)

    # Accuracy computation
    accuracy = compute_accuracy(y_pred, y)

    # Initialize variables and create a saver
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Create session
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            # Shuffle training data
            X_train, Y_train = shuffle_data(X_train, Y_train)

            for i in range(0, m, batch_size):
                X_batch = X_train[i:i+batch_size]
                Y_batch = Y_train[i:i+batch_size]

                # Training step
                _, step_cost, step_accuracy = sess.run(
                    [train_op, cost, accuracy], feed_dict={
                        x: X_batch, y: Y_batch})

                # Print every 100 steps
                if i % 100 == 0:
                    print("\tStep {}: ".format(i))
                    print("\t\tCost: {:.6f}".format(step_cost))
                    print("\t\tAccuracy: {:.6f}".format(step_accuracy))

            # Print after each epoch
            train_cost, train_accuracy = sess.run(
                [cost, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run(
                [cost, accuracy], feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch + 1))
            print("\tTraining Cost: {:.6f}".format(train_cost))
            print("\tTraining Accuracy: {:.6f}".format(train_accuracy))
            print("\tValidation Cost: {:.6f}".format(valid_cost))
            print("\tValidation Accuracy: {:.6f}".format(valid_accuracy))

        # Save the model
        save_path = saver.save(sess, save_path)
        print("Model saved in path: {}".format(save_path))

    return save_path


def initialize_parameters(layers):
    """Init"""
    parameters = {}
    for i in range(1, len(layers)):
        parameters[
            'W' + str(i)] = tf.get_variable('W' + str(i), shape=(
                layers[i-1], layers[i]),
                initializer=tf.contrib.layers.xavier_initializer())
        parameters['b' + str(i)] = tf.get_variable(
            'b' + str(i), shape=(
                1, layers[i]), initializer=tf.zeros_initializer())
    return parameters


def forward_propagation(x, parameters, activations):
    """Forward Prop"""
    a = x
    for i in range(1, len(parameters)//2 + 1):
        z = tf.add(tf.matmul(
            a, parameters['W' + str(i)]), parameters['b' + str(i)])
        if activations[i-1] == 'sigmoid':
            a = tf.nn.sigmoid(z)
        elif activations[i-1] == 'relu':
            a = tf.nn.relu(z)
    return a


def compute_cost(y_pred, y):
    """Compute"""
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
    return cost


def compute_accuracy(y_pred, y):
    """Accuracy"""
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def shuffle_data(X, Y):
    """Shuffle"""
    perm = np.random.permutation(X.shape[0])
    X_shuffled = X[perm]
    Y_shuffled = Y[perm]
    return X_shuffled, Y_shuffled


def create_optimizer(alpha, beta1, beta2, epsilon, decay_rate):
    """Optimizer"""
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.inverse_time_decay(
        alpha, global_step, decay_rate, 1, staircase=True)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    return optimizer
