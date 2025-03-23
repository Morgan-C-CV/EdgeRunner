import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("CUDA:", tf.test.is_gpu_available())
print("GPU:", tf.config.list_physical_devices('GPU'))