import numpy as np

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
ran = np.random.random_sample(input_shape)
input_data = np.array(ran, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

from tensorflow.keras import layers,Sequential
from tensorflow.keras.models import Model
# Adding custom layers
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1024, activation="relu")(x)
predictions = layers.Dense(num_class, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

interpreter.invoke()

# Compile the model
print('Compiling Model')
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
