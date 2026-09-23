import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2


def load_model():

    model = MobileNetV2(
        weights="imagenet",
        include_top=True
    )

    return model