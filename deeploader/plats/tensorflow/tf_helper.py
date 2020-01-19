# coding:utf-8
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import sys
import time

def print_vars(tag, list):
    print("%s:%d" % (tag, len(list)))
    for var in list:
        print('\tname:{0} shape:{1}'.format(var.name, var.shape))


def get_latest_checkpoint(path):
    dir = os.path.dirname(path)
    name = os.path.basename(path)
    latest = tf.train.latest_checkpoint(dir)
    if latest is None:
        return None

    latest_name = os.path.basename(latest)
    print("latest checkpoint:%s" % latest)

    if latest_name.startswith(name):
        return latest
    return None


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, v in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def name_to_scope(name):
    segs = name.split(':')
    return segs[0]


def get_var_list(filter=None):
    learn_vars = tf.trainable_variables()
    if not filter:
        return learn_vars
    var_list = []
    for var in learn_vars:
        if filter(var):
            var_list.append(var)
    return var_list


def get_tensor_from_tower(name, tower_id=0):
    ret = None

    # try input split
    try:
        tname = '%s_split:%d' % (name, tower_id)
        ret = tf.get_default_graph().get_tensor_by_name(tname)
        return ret
    except:
        # print('Tensor %s not in graph' % tname)
        pass

    # try tower
    tname = 'tower%d/%s' % (tower_id, name)
    if name.find(':') >= 0:
        ret = tf.get_default_graph().get_tensor_by_name(tname)
    else:
        tname = tname + ':0'
        ret = tf.get_default_graph().get_tensor_by_name(tname)
    return ret


def build_mgpu_model(opt, gpus, input_holders, builder, *args, **kargs):
    num_gpus = len(gpus)
    with tf.device('/cpu:0'):
        # split input
        split_holders = []
        for input in input_holders:
            # with tf.name_scope(name_to_scope(input.name)) as scope:
            name = name_to_scope(input.name) + '_split'
            split = tf.split(input, num_or_size_splits=num_gpus, axis=0, name=name)
            split_holders.append(split)
        # group    
        holders = []
        for i in range(num_gpus):
            tower_holders = []
            for input in split_holders:
                tower_holders.append(input[i])
            holders.append(tower_holders)

    # Var filter
    var_filter = None
    if 'var_filter' in kargs:
        var_filter = kargs['var_filter']

    # Calculate the gradients for each model tower.
    tower_grads = []
    bundles = []
    loss_sum = 0
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        for i in range(num_gpus):
            gpu_id = gpus[i]
            with tf.device('/gpu:%d' % gpu_id):
                with tf.name_scope('tower%d' % i):
                    bundle = builder(holders[i], gpu_id, *args, **kargs)
                    loss = bundle['loss']
                    # apply var filter
                    var_list = get_var_list(var_filter)
                    # print(var_list)
                    grads = opt.compute_gradients(loss, var_list=var_list)
                    tower_grads.append(grads)
                    bundles.append(bundle)
                    loss_sum += loss

    loss_ave = loss_sum / num_gpus
    grads = average_gradients(tower_grads)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # Apply the gradients to adjust the shared variables.
        train_op = opt.apply_gradients(grads)

    return train_op, loss_ave, bundles


def build_mgpu_model_grad_accumulation(opt, gpus, steps, input_holders, builder, *args, **kargs):
    num_gpus = len(gpus)
    with tf.device('/cpu:0'):
        # split input
        split_holders = []
        for input in input_holders:
            # with tf.name_scope(name_to_scope(input.name)) as scope:
            name = name_to_scope(input.name) + '_split'
            split = tf.split(input, num_or_size_splits=num_gpus, axis=0, name=name)
            split_holders.append(split)
        # group
        holders = []
        for i in range(num_gpus):
            tower_holders = []
            for input in split_holders:
                tower_holders.append(input[i])
            holders.append(tower_holders)

    # Var filter
    var_filter = None
    if 'var_filter' in kargs:
        var_filter = kargs['var_filter']

    # Calculate the gradients for each model tower.
    tower_grads = []
    bundles = []
    loss_sum = 0
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        for i in range(num_gpus):
            gpu_id = gpus[i]
            with tf.device('/gpu:%d' % gpu_id):
                with tf.name_scope('tower%d' % i):
                    bundle = builder(holders[i], gpu_id, *args, **kargs)
                    loss = bundle['loss']
                    # apply var filter
                    var_list = get_var_list(var_filter)
                    # print(var_list)
                    grads = opt.compute_gradients(loss, var_list=var_list)
                    tower_grads.append(grads)
                    bundles.append(bundle)
                    loss_sum += loss

    loss_ave = loss_sum / num_gpus
    grads = average_gradients(tower_grads)

    # grad accumulation
    trainable_vars = get_var_list(var_filter)
    # Create variables to store accumulated gradients
    accumulators = [
        tf.Variable(
            tf.zeros_like(tv.initialized_value()),
            trainable=False
        ) for tv in trainable_vars
    ]

    # Create a variable for counting the number of accumulations
    accumulation_counter = tf.Variable(0.0, trainable=False)

    # Compute gradients; grad_pairs contains (gradient, variable) pairs
    grad_pairs = grads

    # Create operations which add a variable's gradient to its accumulator.
    accumulate_ops = [
        accumulator.assign_add(
            grad
        ) for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)
    ]

    # The final accumulation operation is to increment the counter
    accumulate_ops.append(accumulation_counter.assign_add(1.0))

    # Update trainable variables by applying the accumulated gradients
    # divided by the counter. Note: apply_gradients takes in a list of
    # (grad, var) pairs
    with tf.control_dependencies([accumulate_ops]):
        train_step = opt.apply_gradients(
            [(accumulator / accumulation_counter, var)
             for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)]
        )

    with tf.control_dependencies([train_step]):
        # Accumulators must be zeroed once the accumulated gradient is applied.
        zero_ops = [
            accumulator.assign(
                tf.zeros_like(tv)
            ) for (accumulator, tv) in zip(accumulators, trainable_vars)
        ]

        # Add one last op for zeroing the counter
        zero_ops.append(accumulation_counter.assign(0.0))

    last_op = tf.cond(tf.less(accumulation_counter, steps), accumulate_ops, zero_ops)
    return last_op, loss_ave, bundles


def create_session(allow_growth=True, per_gpu_memory_fraction=0):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if per_gpu_memory_fraction > 0:
        config.gpu_options.per_process_gpu_memory_fraction = per_gpu_memory_fraction
    session = tf.Session(config=config)
    return session


def restore_from_checkpoint(sess, ckpt, var_list=None):
    if os.path.isdir(ckpt):
        latest_checkpoint = get_latest_checkpoint(ckpt)
    else:
        latest_checkpoint = ckpt
    if not var_list:
        var_list = tf.trainable_variables()
    print('restore from:%s' % (latest_checkpoint))
    if latest_checkpoint is not None:
        _fn = slim.assign_from_checkpoint_fn(latest_checkpoint, var_list=var_list)
        _fn(sess)
    return True if latest_checkpoint else False
