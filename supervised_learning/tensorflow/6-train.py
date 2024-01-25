#!/usr/bin/env python3
"""Training Operation"""
import tensorflow as tf
calculate_accuracy = __import__(
    '3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__(
    '0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        layer_sizes,
        activations,
        alpha,
        iterations,
        save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.

    Arguments:
    X_train: numpy.ndarray containing the training input data
    Y_train: numpy.ndarray containing the training labels
    X_valid: numpy.ndarray containing the validation input data
    Y_valid: numpy.ndarray containing the validation labels
    layer_sizes: the number of nodes in each layer of the network
    activations: the activation functions for each layer of the network
    alpha: learning rate
    iterations: number of iterations to train over
    save_path: path to save the model

    Returns:
    the path where the model was saved
    """
    # Reset the default graph to ensure a clean slate
    tf.reset_default_graph()

    # Create placeholders for input data and labels
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # Build the forward propagation graph
    y_pred = forward_prop(x, layer_sizes, activations)

    # Calculate loss and accuracy
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    # Create the training operation
    train_op = create_train_op(loss, alpha)

    # Add tensors and operation to the graph collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    # Initialize variables
    init = tf.global_variables_initializer()

    # Start a session
    with tf.Session() as sess:
        sess.run(init)

        # Training loop
        for i in range(iterations + 1):
            # Train the model
            _, train_cost, train_accuracy = sess.run(
                [train_op, loss, accuracy],
                feed_dict={x: X_train, y: Y_train})

            # Print training progress
            if i % 100 == 0 or i == 0 or i == iterations:
                valid_cost, valid_accuracy = sess.run(
                    [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_accuracy))

        # Save the trained model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_path)

    return save_path
