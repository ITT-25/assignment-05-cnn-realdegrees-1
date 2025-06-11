import json
import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, RandomFlip, BatchNormalization, Input
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import load_model
from typing import Optional, Tuple
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import threading
import queue
from keras.metrics import categorical_crossentropy

IMG_SIZE = 64
SIZE = (IMG_SIZE, IMG_SIZE)

class GestureModel:
    def __init__(self, color_channels: int, dataset_path: str, gesture_actions: dict):
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gesture_model.keras")
        self.size = SIZE
        self.color_channels = color_channels
        self.dataset_path = dataset_path
        self.gesture_actions = gesture_actions
        self.model = None
        self.label_names = []
        self.img: np.ndarray = None
        self.prediction_queue = queue.Queue(maxsize=1)
        self._latest_prediction: Tuple[str, float] = None

        # Load existing model or train a new one
        self._load()

        # Start prediction thread
        self._stop_event = threading.Event()
        self._prediction_thread = threading.Thread(target=self._prediction_loop, daemon=True)
        self._prediction_thread.start()

    def _load(self):
        if os.path.exists(self.model_path):
            print(f"Loading saved model from {self.model_path}...")
            self.model = load_model(self.model_path)
            _, _, _, _, self.label_names = self.load_dataset(self.dataset_path, self.gesture_actions)
        else:
            print("No existing model found, training a new one...")
            print("Loading and preparing dataset...")
            X_train, X_test, y_train, y_test, self.label_names = self.load_dataset(
                self.dataset_path, self.gesture_actions
            )
            print("Building and training model...")
            self.model = self.build_model(X_train, X_test, y_train, y_test)

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        img_resized = cv2.resize(img, self.size)
        if self.color_channels == 1:
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        return img_gray

    def load_dataset(
        self, dataset_path: str, gesture_actions: dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        images = []
        labels = []
        label_names = []
        for condition in gesture_actions.keys():
            with open(f"{dataset_path}/_annotations/{condition}.json") as f:
                annotations = json.load(f)
            for filename in os.listdir(f"{dataset_path}/{condition}"):
                UID = filename.split(".")[0]
                img = cv2.imread(f"{dataset_path}/{condition}/{filename}")
                if img is None or UID not in annotations:
                    continue
                annotation = annotations[UID]
                for i, bbox in enumerate(annotation["bboxes"]):
                    x1 = int(bbox[0] * img.shape[1])
                    y1 = int(bbox[1] * img.shape[0])
                    w = int(bbox[2] * img.shape[1])
                    h = int(bbox[3] * img.shape[0])
                    x2 = x1 + w
                    y2 = y1 + h
                    crop = img[y1:y2, x1:x2]
                    preprocessed = self.preprocess_image(crop)
                    label = annotation["labels"][i]
                    if label not in label_names:
                        label_names.append(label)
                    label_index = label_names.index(label)
                    images.append(preprocessed)
                    labels.append(label_index)
        X = np.array(images).astype("float32") / 255.0
        y = np.array(labels)
        X = X.reshape(-1, self.size[0], self.size[1], self.color_channels)
        y_one_hot = to_categorical(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, label_names

    def build_model(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
        batch_size = 32
        epochs = 100
        num_classes = y_train.shape[1]
        input_shape = X_train.shape[1:]  # Get input shape from data

        # Use LeakyReLU for conv layers, ReLU for dense
        activation = "relu"
        activation_conv = "leaky_relu"

        model = Sequential()

        # Input layer + Data Augmentation
        model.add(Input(shape=input_shape))
        model.add(RandomFlip("horizontal"))

        # Convolutional Blocks
        # Block 1
        model.add(Conv2D(32, (3, 3), activation=activation_conv, padding="same"))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), activation=activation_conv, padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block 2
        model.add(Conv2D(64, (3, 3), activation=activation_conv, padding="same"))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation=activation_conv, padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block 3
        model.add(Conv2D(128, (3, 3), activation=activation_conv, padding="same"))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), activation=activation_conv, padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        # Block 4
        model.add(Conv2D(256, (3, 3), activation=activation_conv, padding="same"))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3, 3), activation=activation_conv, padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(512, activation=activation))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(256, activation=activation))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Output layer
        model.add(Dense(num_classes, activation="softmax"))

        # Optimizer with lower learning rate
        model.compile(loss=categorical_crossentropy, optimizer="adam", metrics=["accuracy"])

        # Callbacks
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1)

        stop_early = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)

        print("Training model...")
        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_test),
            callbacks=[reduce_lr, stop_early],
        )

        print(f"Saving model to {self.model_path}...")
        model.save(self.model_path)
        return model

    def set_img(self, img: np.ndarray):
        self.img = img

    def _prediction_loop(self):
        while not self._stop_event.is_set():
            if self.img is not None and self.model is not None:
                hand_img = np.expand_dims(self.img, axis=0) / 255.0
                prediction = self.model.predict(hand_img, verbose=0)
                label_index = int(np.argmax(prediction))
                confidence = float(prediction[0][label_index] * 100)
                result = (self.label_names[label_index], confidence)
                self._latest_prediction = result
                # Replace old prediction if queue is full
                if self.prediction_queue.full():
                    try:
                        self.prediction_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.prediction_queue.put(result)
                self.img = None  # Clear the image after processing
            else:
                threading.Event().wait(0.01)

    def get_prediction(self, img: Optional[np.ndarray] = None) -> Optional[Tuple[int, float]]:
        if img is not None:
            self.set_img(img)
        return self._latest_prediction
