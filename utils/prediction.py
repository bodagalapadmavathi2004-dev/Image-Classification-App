import numpy as np

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions
)


def prepare_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image to MobileNetV2 input size
    image = image.resize((224, 224))

    # Convert image to NumPy array
    image_array = np.array(image)

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Apply MobileNetV2 preprocessing
    image_array = preprocess_input(
        image_array
    )

    return image_array


def predict_image(model, image):

    # Prepare image
    processed_image = prepare_image(
        image
    )

    # Generate predictions
    predictions = model.predict(
        processed_image,
        verbose=0
    )

    # Decode Top 5 ImageNet predictions
    decoded_predictions = decode_predictions(
        predictions,
        top=5
    )[0]

    results = []

    for _, label, confidence in decoded_predictions:

        results.append(
            {
                "label": label.replace(
                    "_",
                    " "
                ).title(),

                "confidence": float(
                    confidence
                )
            }
        )

    return results