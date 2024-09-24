sns.set(style="whitegrid", palette="muted", font_scale=1.5)
RANDOM_SEED = 42
 
from google.colab import drive
drive.mount('/content/drive')

from google.colab import files
uploaded = files.upload()

#transforming shape
reshaped_segments = np.asarray(
	segments, dtype = np.float32).reshape(
	-1 , N_time_steps, N_features)

reshaped_segments.shape

## TRAIN_TEST_SPLIT

X_train, X_test, Y_train, Y_test = train_test_split(
	reshaped_segments, labels, test_size = 0.2, 
	random_state = RANDOM_SEED)

##Model building

def create_LSTM_model(inputs):
	W = {
		'hidden': tf.Variable(tf.random_normal([N_features, N_hidden_units])),
		'output': tf.Variable(tf.random_normal([N_hidden_units, N_classes]))
	}
	biases = {
		'hidden': tf.Variable(tf.random_normal([N_hidden_units], mean = 0.1)),
		'output': tf.Variable(tf.random_normal([N_classes]))
	}
	X = tf.transpose(inputs, [1, 0, 2])
	X = tf.reshape(X, [-1, N_features])
	hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])
	hidden = tf.split(hidden, N_time_steps, 0)

	lstm_layers = [tf.contrib.rnn.BasicLSTMCell(
		N_hidden_units, forget_bias = 1.0) for _ in range(2)]
	lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)

	outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, 
										hidden, dtype = tf.float32)

	lstm_last_output = outputs[-1]
	return tf.matmul(lstm_last_output, W['output']) + biases['output']


