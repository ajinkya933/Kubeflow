import kfp
import sys
import kfp.dsl as dsl
import kfp.components as comp


def train(data_path, model_file):
    '''
        This definition contains MNIST training steps:
            * Data Import
            * Data Preprocessing
            * Keras model creation
            * Model optimizer : adam
            * Training with specified epoch
            * Print test accuracy
            * Save the model
    '''
    import pickle
    import tensorflow as tf
    from tensorflow.python import keras

    # Load dataset
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Normalize dataset
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Define the model using keras
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)

    ])

    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(
        from_logits=True), metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('Test accuracy', test_acc)

    # IMP this specifies the path inside the Docker container where our model will be saved
    model.save(f'{data_path}/{model_file}')

    # Save test data pickle file
    with open(f'{data_path}/test_data', 'wb') as f:
        pickle.dump((test_images, test_labels), f)


# Glue together training function to the docker container
train_op = comp.func_to_container_op(train, base_image='tensorflow/tensorflow:latest-gpu-py3')


# define pipeline metadata like name, description etc.
@dsl.pipeline(
    name='MNIST Pipeline',
    description='Pipeline that trains MNIST models on GPU'
)
# define virtual HDD space that the pipeline will take to run
def mnist_container_pipeline(data_path='/mnt', model_file='mnist_model.h5'):
    vop = dsl.VolumeOp(
        name='create_volume',
        resource_name='data-volume',
        size='1Gi',
        modes=dsl.VOLUME_MODE_RWM
    )

# We have already created a volume and a Glued component(Docker+Python script). This Glued component needs 
# to communicate with the volume so lets attach a volume to Glued component 
    mnist_training_container = train_op(data_path, model_file) \
        .add_pvolumes({data_path: vop.volume})



# pass these paths inside the volume that we made
pipeline_func = mnist_container_pipeline
#experiment_name = 'fashion_mnist_kubeflow_training2'
experiment_name = sys.argv[0]
run_name = pipeline_func.__name__ + ' run'


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(pipeline_func, __file__ + '.yaml')




