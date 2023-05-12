import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers,models
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping,ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model
from tensorflow.keras.layers.experimental import preprocessing
from pathlib import Path
import os.path

from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, compare_historys, walk_through_dir, pred_and_plot

# Metrics
from sklearn.metrics import classification_report, confusion_matrix
import itertools

def make_model():
    BATCH_SIZE = 32
    IMAGE_SIZE = (320, 320)
    dataset = "/content/drive/MyDrive/Orbit/uprak-deployment/uprak_deployment/train/fire_dataset"
    
    image_dir = Path(dataset)

    # Get filepaths dan labels
    filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png'))

    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Gabungkan filepaths and labels
    image_df = pd.concat([filepaths, labels], axis=1)
    # pisahkan train dan test data
    train_df, test_df = train_test_split(image_df, test_size=0.2, shuffle=True, random_state=42)

    train_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
    )

    test_generator = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )
    # Split the data menjadi 3 kategori.
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=42,
        subset='training'
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=42,
        subset='validation'
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )
    # Resize Layer
    resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(224,224),
    layers.experimental.preprocessing.Rescaling(1./255),
    ])
    # Load the pretained model
    pretrained_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    pretrained_model.trainable = False

    # Create checkpoint callback
    checkpoint_path = "fires_classification_model_checkpoint"
    checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                        save_weights_only=True,
                                        monitor="val_accuracy",
                                        save_best_only=True)
    
    # Setup EarlyStopping callback untuk menghentikan proses training jika val_loss pada model tidak bisa muncul sampai 3 epochs
    early_stopping = EarlyStopping(monitor = "val_loss", # metric val loss
                                patience = 5,
                                restore_best_weights = True) # jika val_loss mengurangi 3 epoch berturut-turut, maka stop training
    
    inputs = pretrained_model.input
    x = resize_and_rescale(inputs)

    x = Dense(256, activation='relu')(pretrained_model.output)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)


    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_images,
        steps_per_epoch=len(train_images),
        validation_data=val_images,
        validation_steps=len(val_images),
        epochs=100,
        callbacks=[
            early_stopping,
            create_tensorboard_callback("training_logs", 
                                        "fire_classification"),
            checkpoint_callback,
        ]
    )

    return model
