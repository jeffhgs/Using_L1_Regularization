import json
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def setup(options):
    if options.regL1Try3 is not None:
        # Work around GPU bug in TF 2.3
        #     tensorflow.python.framework.errors_impl.NotFoundError: No registered 'ResourceApplyProximalAdagrad' OpKernel for 'GPU' devices compatible with node {{node ResourceApplyProximalAdagrad}}
        #     . . .
        # https://github.com/tensorflow/tensorflow/commit/a1b64cf2a6a995ffaaf384cf8643221f1c27db48
        try:
            # Disable all GPUS
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != 'GPU'
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

epsilon_zero=0.00001

def count_nonzero_model_weights(model):
    vars = {}
    numNzTotal = 0
    numTotal = 0
    for rv in model.trainable_variables:
        # rv is a tensorflow.python.ops.resource_variable_ops.ResourceVariable
        name = rv.name
        (isNz, numOf) = np.unique(np.fabs(rv.value().numpy())>epsilon_zero, return_counts=True)
        d = dict(zip(isNz, numOf))
        num = int((d[False] if False in d else 0) + (d[True] if True in d else 0))
        numNz = int(d[True] if True in d else 0)
        numNzTotal += numNz
        numTotal += num
        vars[name] = {"num":num, "numNz":numNz}
    vars["total"] = {"num":numTotal, "numNz":numNzTotal}
    return vars



def proximalUpdate(model, learningRate, cL1):
    train_vars = model.trainable_variables
    for rv in train_vars:
        # rv.__class__ is tensorflow.python.ops.resource_variable_ops.ResourceVariable
        v=rv.value()
        v2=np.sign(v)*np.fmax(np.fabs(v) - learningRate*cL1,0)
        rv.assign(v2)


def run_train(options):
    model, outputs, x1, x2 = build_model(options)

    # Instantiate an optimizer.
    learningRate = 1e-3
    if(options.regL1Try2 is not None or options.regL1Try3 is not None):
        cL1=options.regL1Try3 if options.regL1Try3 is not None else options.regL1Try2
        optimizer = tf.compat.v1.train.ProximalAdagradOptimizer(
            learningRate,
            initial_accumulator_value=learningRate**2,
            l1_regularization_strength=cL1)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=learningRate)
    # Instantiate a loss function.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric_fn = keras.metrics.SparseCategoricalAccuracy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[metric_fn])

    # Prepare the training dataset.
    batch_size = 64
    x_train, x_test, y_train, y_test = setup_mnist_data(options.numSamples)

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(batch_size)

    for epoch in range(options.numEpochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                logits = model(x_batch_train, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            if (options.regL1Alt is not None):
                proximalUpdate(model, learningRate, options.regL1Alt)

#            if step % 200 == 0:
        results_train = model.evaluate(x_train, y_train, verbose=0)
        results_test = model.evaluate(x_test, y_test, verbose=0)
        nzByVar = count_nonzero_model_weights(model)
        print(json.dumps({"event": "measure_test",
                          "epochAfter": epoch,
                          "ent_train": results_train[0],
                          "inacc_train": 1-results_train[1],
                          "ent_test": results_test[0],
                          "inacc_test": 1-results_test[1],
                          "num_nonzero": nzByVar
                          }),
              flush=True)


def setup_mnist_data(numSamples=10000):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (-1, 784))
    # Reserve 10,000 samples for validation.
    x_test = x_train[-10000:][0:numSamples]
    y_test = y_train[-10000:][0:numSamples]
    x_train = x_train[:-10000][0:5 * numSamples]
    y_train = y_train[:-10000][0:5 * numSamples]
    return x_train, x_test, y_train, y_test


def build_model(options):
    inputs = keras.Input(shape=(784,), name="digits")
    x1 = None
    x2 = None
    outputs = None
    if options.regL1Try1 is not None:
        c = options.regL1Try1
        x1 = layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l1(l=c))(inputs)
        x2 = layers.Dense(64, activation="relu")(x1)
        outputs = layers.Dense(10, name="predictions")(x2)
    if (options.regL1Try3 is not None or options.regL1Alt is not None):
        x1 = layers.Dense(64, activation="relu")(inputs)
        x2 = layers.Dense(64, activation="relu")(x1)
        outputs = layers.Dense(10, name="predictions")(x2)
    else:
        x1 = layers.Dense(64, activation="relu")(inputs)
        x2 = layers.Dense(64, activation="relu")(x1)
        outputs = layers.Dense(10, name="predictions")(x2)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model, outputs, x1, x2


def parse_options():
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--regL1Try1', type=float, default=None)
    parser.add_argument('--regL1Try2', type=float, default=None)
    parser.add_argument('--regL1Try3', type=float, default=None)
    parser.add_argument('--regL1Alt', type=float, default=None)
    parser.add_argument('--numSamples', type=int, default=10000)
    parser.add_argument('--numEpochs', type=int, default=20)
    options = parser.parse_args(sys.argv[1:])
    return options

if __name__ == '__main__':
    options = parse_options()
    setup(options)
    print(json.dumps({"event": "args",
                      "args": options.__dict__,
                      "secStarted": time.time(),
                      "timeStarted": time.asctime(time.gmtime())}))
    run_train(options)

