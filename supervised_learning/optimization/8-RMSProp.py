#!/usr/bin/env python3
"""
creates the training operation for a neural network
in tensorflow using the RMSProp optimization algorithm
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    creates the training operation for a neural network
    in tf using the RMSProp op.

    args:
        loss: the loss of the network
        alpha: the learning rate
        beta2: the RMSProp weight
        epsilon: a small number to avoid division by zero

    returns:
        the RMSProp optimization operation
    """
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=alpha, decay=beta2, epsilon=epsilon
        )
    train_op = optimizer.minimize(loss)
    return train_op
