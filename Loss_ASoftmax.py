import tensorflow as tf

def Loss_ASoftmax(x, y, l, num_cls, m = 2, name = 'asoftmax'):
    '''
    x: B x D - data
    y: B x 1 - label
    l: 1 - lambda 
    '''
    xs = x.get_shape()
    w = tf.get_variable("asoftmax/W", [xs[1], num_cls], dtype=tf.float32, 
            initializer=tf.contrib.layers.xavier_initializer())

    eps = 1e-8

    xw = tf.matmul(x,w) 

    if m == 0:
        return xw, tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=xw))

    w_norm = tf.norm(w, axis = 0) + eps
    logits = xw/w_norm

    if y is None: 
        return logits, None

    ordinal = tf.constant(list(range(0, xs[0])), tf.int64)
    ordinal_y = tf.stack([ordinal, y], axis = 1)

    x_norm = tf.norm(x, axis = 1) + eps

    sel_logits = tf.gather_nd(logits, ordinal_y)

    cos_th = tf.div(sel_logits, x_norm)

    if m == 1:

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

    else:

        if m == 2:

            cos_sign = tf.sign(cos_th)
            res = 2*tf.multiply(tf.sign(cos_th), tf.square(cos_th)) - 1

        elif m == 4:

            cos_th2 = tf.square(cos_th)
            cos_th4 = tf.pow(cos_th, 4)
            sign0 = tf.sign(cos_th)
            sign3 = tf.multiply(tf.sign(2*cos_th2 - 1), sign0)
            sign4 = 2*sign0 + sign3 - 3
            res = sign3*(8*cos_th4 - 8*cos_th2 + 1) + sign4
        else:
            raise ValueError('unsupported value of m')

        scaled_logits = tf.multiply(res, x_norm)

        f = 1.0/(1.0+l)
        ff = 1.0 - f
        comb_logits_diff = tf.add(logits, tf.scatter_nd(ordinal_y, tf.subtract(scaled_logits, sel_logits), logits.get_shape())) 
        updated_logits = ff*logits + f*comb_logits_diff

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=updated_logits))

    return logits, loss
