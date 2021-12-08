import pandas as pd
from tensorflow import keras
from video_data_generator import VideoDataGenerator

# Define hyperparameters
IMG_SIZE = 224
STEP = 4
BATCH_SIZE = 4
EPOCHS = 500
MAX_SEQ_LENGTH = 120
NUM_FEATURES = 128
DATA_PATH = "data"
NUM_CLASSES = 3
FILE_COL = "video_name"
Y_COL = "tag"
CLASS_MAPPING = {"follow_up": 0, "at": 1, "referral": 2}

# Data preparation
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

print(f"Total number of videos for training: {len(train_df)}")
print(f"Total number of videos for validation: {len(val_df)}")
print(f"Total number of videos for testing: {len(test_df)}")

train_data_gen = VideoDataGenerator(
    train_df,
    DATA_PATH,
    BATCH_SIZE,
    file_col=FILE_COL,
    y_col=Y_COL,
    mapping=CLASS_MAPPING,
    max_frames=MAX_SEQ_LENGTH,
    img_size=IMG_SIZE,
    step=STEP
)

val_data_gen = VideoDataGenerator(
    val_df,
    DATA_PATH,
    BATCH_SIZE,
    file_col=FILE_COL,
    y_col=Y_COL,
    mapping=CLASS_MAPPING,
    max_frames=MAX_SEQ_LENGTH,
    img_size=IMG_SIZE,
    step=STEP
)

test_data_gen = VideoDataGenerator(
    test_df,
    DATA_PATH,
    BATCH_SIZE,
    file_col=FILE_COL,
    y_col=Y_COL,
    mapping=CLASS_MAPPING,
    max_frames=MAX_SEQ_LENGTH,
    img_size=IMG_SIZE,
    step=STEP
)

# building the model
def build_model():
    cnn = keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    input_layer = keras.Input((MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    model = keras.layers.TimeDistributed(cnn)(input_layer, mask=mask_input)
    model = keras.layers.GRU(NUM_FEATURES)(model)
    output = keras.layers.Dense(NUM_CLASSES, activation="softmax")(model)

    model = keras.Model([input_layer, mask_input], output)

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model

def run_experiment():
    checkpoint_path = "checkpoints/"
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_weights_only=True, save_best_only=True, verbose=1
    )

    model = build_model()
    history = model.fit_generator(
        train_data_gen,
        validation_data=val_data_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    return history, model

if __name__ == "__main__":
    hist, model = run_experiment()
    pd.DataFrame(hist).to_csv("history.csv", index=False)
    _, accuracy = model.evaluate(test_data_gen)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
