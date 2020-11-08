from tensorflow.keras.preprocessing import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dropout,Flatten,Dense,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizer import SGD,Adam

def build_model(nbr_classes):

    base_model = InceptionV3(weights="imagenet",include_top=False,input_tensor=Input(shape=(229,229,3)))

    head_model = base_model.output

    head_model = Flatten()(head_model)
    head_model = Dense(512)(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(nbr_classes, activation="softmax")(head_model)

    model = Model(inputs=base_model.input,outputs=head_model)

    for layer in base_model.layers:
        layer.trainable = False

    return model
