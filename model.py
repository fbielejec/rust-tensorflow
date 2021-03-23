import tensorflow as tf
import tensorflow_probability as tfp

# --- model

@tf.function (input_signature=(tf.TensorSpec(shape=(None,), dtype=tf.float32),
                               tf.TensorSpec(shape=(None,), dtype=tf.float32)))
def cor_na_omit (x, y):
    ind1 = tf.math.logical_not (tf.math.is_nan(x))
    ind2 = tf.math.logical_not (tf.math.is_nan(y))
    indices = tf.reduce_all(tf.stack ([ind1, ind2]), 0)
    return tfp.stats.correlation (x [indices], y [indices], sample_axis = 0, event_axis = None)

class Recommender(tf.Module):

  def __init__(self):
    super(Recommender, self).__init__()

  @tf.function (input_signature=(tf.TensorSpec(shape=[None,None], dtype=tf.float32),
                                 tf.TensorSpec(shape=(), dtype=tf.int32)))
  def __call__(self, scores, user_index):
    tf.print('Executing with tensorflow v' + tf.__version__)
    user_row = tf.reshape(tf.gather(scores, user_index), [-1])
    c_hat = tf.map_fn(fn=lambda row: cor_na_omit (user_row, row), elems = scores)
    # keep only positively correlated users
    users_indices = tf.where(c_hat > 0)[:,-1]
    # remove user row
    not_user_index = tf.where (users_indices != tf.cast(user_index, dtype=tf.int64))[:,-1]
    users_indices = tf.gather (users_indices, not_user_index, axis=0)
    # only videos user has not seen
    videos_indices = tf.where(tf.math.is_nan(user_row))[:,-1]
    c_hat = tf.gather (c_hat, users_indices, axis=0)
    s = tf.gather (tf.gather (scores, users_indices, axis=0), videos_indices, axis=1)
    # fill missing values with 0's
    s = tf.where(tf.math.is_nan(s), tf.zeros_like(s), s)
    c_hat = tf.reshape (c_hat, [1, tf.shape (c_hat) [0]])
    r = tf.matmul (c_hat, s)
    r = r[-1]
    # remove 0 and NAN scored recommendations
    non_zero_indices = tf.where(r > 0)[:,-1]
    r = tf.gather (r, non_zero_indices, axis=0)
    total = tf.math.reduce_sum (c_hat)
    r = r/total
    order = tf.argsort(r, direction='DESCENDING', axis=0)
    videos_indices = tf.cast(videos_indices, dtype=tf.float32)
    recommendations = tf.stack([tf.gather (videos_indices, order),
                                tf.gather (r, order)],
                               axis=1)
    return recommendations

# --- save model

nan = float('NaN')
scores = tf.constant([
    # v1   v2   v3   v4   v5   v6
    [ 2.5, 3.5, 3.0, 3.5, 2.5, 3.0 ], # u1
    [ 3.0, 3.5, 1.5, 5.0, 3.5, 3.0 ], # u2
    [ 2.5, 3.0, nan, 3.5, nan, 4.0 ], # u3
    [ nan, 3.5, 3.0, 4.0, 2.5, 4.5 ], # u4
    [ 3.0, 4.0, 2.0, 3.0, 2.0, 3.0 ], # u5
    [ 3.0, 4.0, nan, 5.0, 3.5, 3.0 ], # u6
    [ nan, 4.5, nan, 4.0, 1.0, nan ]  # u7
])

# saved_model_cli show --dir ./saved_model --all
PATH = "./saved_model"
model = Recommender ()
# print (model (scores, tf.constant(6)))
tf.saved_model.save(model, PATH)

# loaded = tf.keras.models.load_model(PATH)
# print (loaded (scores, tf.constant(6)))
