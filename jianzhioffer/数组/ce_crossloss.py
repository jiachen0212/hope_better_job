# np.nan_to_num: 使用0代替数组x中的nan元素，使用有限的数字代替inf元素
def cross_entropy(a, y):
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

# tensorflow version
loss = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

# numpy version
loss = np.mean(-np.sum(y_*np.log(y), axis=1))