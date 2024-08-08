
import tensorflow as tf
import tensorflow_transform as tft

# Define the numerical features that will be used in the model.
NUMERICAL_FEATURES = ['trip_miles', 'trip_seconds']

# Define the features that will be bucketized.
BUCKET_FEATURES = [
    'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
    'dropoff_longitude'
]

# Define the number of buckets used by tf.transform for encoding each feature in BUCKET_FEATURES.
FEATURE_BUCKET_COUNT = 10

# Define the categorical features that are represented as numerical values.
CATEGORICAL_NUMERICAL_FEATURES = [
    'trip_start_hour', 'trip_start_day', 'trip_start_month',
    'pickup_census_tract', 'dropoff_census_tract', 'pickup_community_area',
    'dropoff_community_area'
]

# Define the categorical features that are represented as strings.
CATEGORICAL_STRING_FEATURES = [
    'payment_type',
    'company',
]

# Define the number of vocabulary terms used for encoding categorical features.
VOCAB_SIZE = 1000

# Define the count of out-of-vocab buckets in which unrecognized categorical are hashed.
OOV_SIZE = 10

# Define the keys for the label and fare columns in the input data.
LABEL_KEY = 'fare'

# Define a helper function that appends the suffix '_xf' to a feature key to avoid clashes
# with raw feature keys when running the Evaluator component.
def t_name(key):
    return key + '_xf'

def _make_one_hot(x, key):
    """Make a one-hot tensor to encode categorical features.
    Args:
        x: A dense tensor
        key: A string key for the feature in the input
    Returns:
        A dense one-hot tensor as a float list
    """
    # Computing and applying vocabulary to the input tensor and integerizing it.
    integerized = tft.compute_and_apply_vocabulary(x,
                                                   top_k=VOCAB_SIZE,
                                                   num_oov_buckets=OOV_SIZE,
                                                   vocab_filename=key,
                                                   name=key)
    # Getting the vocabulary size for the feature.
    depth = (
        tft.experimental.get_vocabulary_size_by_name(key) + OOV_SIZE)
    # Converting the integerized tensor to a one-hot tensor.
    one_hot_encoded = tf.one_hot(
        integerized,
        depth=tf.cast(depth, tf.int32),
        on_value=1.0,
        off_value=0.0)
    return tf.reshape(one_hot_encoded, [-1, depth])

def _fill_in_missing(x):
    """Replace missing values in a SparseTensor.
    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
    Args:
      x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
        in the second dimension.
    Returns:
      A rank 1 tensor where missing values of `x` have been filled in.
    """
    if not isinstance(x, tf.sparse.SparseTensor):
        return x

    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)

def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
      inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
      Map from string feature key to transformed feature operations.
    """
    outputs = {}
    for key in NUMERICAL_FEATURES:
        # Filling in missing values and scaling the numerical features to have mean=0 and variance=1.
        outputs[t_name(key)] = tft.scale_to_z_score(
            _fill_in_missing(inputs[key]), name=key)

    for key in BUCKET_FEATURES:
        # Filling in missing values and bucketizing the features.
        outputs[t_name(key)] = tf.cast(tft.bucketize(
            _fill_in_missing(inputs[key]), FEATURE_BUCKET_COUNT, name=key),
            dtype=tf.float32)

    for key in CATEGORICAL_STRING_FEATURES:
        # Filling in missing values and one-hot encoding the categorical string features.
        outputs[t_name(key)] = _make_one_hot(_fill_in_missing(inputs[key]), key)

    for key in CATEGORICAL_NUMERICAL_FEATURES:
        # Filling in missing values, converting the categorical numerical features to strings, and one-hot encoding them.
        outputs[t_name(key)] = _make_one_hot(tf.strings.strip(
        tf.strings.as_string(_fill_in_missing(inputs[key]))), key)

    # Fare is used as a label here
    outputs[LABEL_KEY] = _fill_in_missing(inputs[LABEL_KEY])

    return outputs

