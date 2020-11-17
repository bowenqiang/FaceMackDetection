import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def train_mask_model(dataset, plot, model_name, epochs):

    batch_size = 32
    img_height = 224
    img_width = 224
    print(dataset)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=dataset,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=dataset,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # use buffered prefetching
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )

    num_classes = 2

    model = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # save trained model
    model.save(model_name, save_format="h5")

    # draw the plot
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    plt.savefig(plot)

if __name__ == "__main__":
    # parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default="dataset", help="path to input dataset")
    ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
    ap.add_argument("-m", "--model", type=str, default="face_mask.model", help="path to output face mask detector model")
    ap.add_argument("-e", "--epochs", type=int, default=10, help="epochs for traning")
    args = vars(ap.parse_args())

    train_mask_model(args["dataset"], args["plot"], args["model"], args["epochs"])