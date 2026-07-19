from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import time


train_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=(28, 28),
    color_mode="grayscale",
    batch_size=32,
    label_mode="int",
    shuffle=True
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(28, 28),
    color_mode="grayscale",
    batch_size=32,
    label_mode="int",
    shuffle=False
)

print("Classes:", train_dataset.class_names)


'''we normalize the pixel values by Rescaling which are between 0 and 255 to be between 0 and 1 by 
dividing by 255.0, which helps in faster convergence during training.'''
'''We use the flatten layer to convert the 2D images into 1D vectors, followed by four dense layers.'''
'''We use that (28,28,1) as there are 28x28 pixel images with 1 channel (grayscale). '''

model=Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])
#We use the Adam optimizer and SparseCategoricalCrossentropy loss function for multi-class classification. The model is compiled with accuracy as the evaluation metric.
model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

start_time = time.time()



'''We train the model for 70 epochs, which means the model will see the entire training dataset 70 times.'''
history = model.fit(train_dataset, epochs=70)

test_loss, test_accuracy = model.evaluate(test_dataset)
train_loss, train_accuracy = model.evaluate(train_dataset)

end_time = time.time()
print("Training time:", end_time - start_time)

print("Train loss:", train_loss*100)
print("Train accuracy:", train_accuracy*100)

print("Test loss:", test_loss*100)
print("Test accuracy:", test_accuracy*100)

'''Save the model so it can be used later for inference or further training.'''
model.save('fashion_classifier.keras')

