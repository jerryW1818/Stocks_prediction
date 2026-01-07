"""Deep LSTM model builders for the Tesla notebook.

Provide a ready-to-use function `build_deep_lstm` that returns a compiled
Keras model. Import and call this from the notebook to replace the existing
`model_seq` construction.
"""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam


def build_deep_lstm(input_shape, lr=1e-3):
    """Build a deeper Bidirectional LSTM regression model.

    Args:
        input_shape (tuple): (timesteps, n_features)
        lr (float): learning rate for Adam optimizer

    Returns:
        tf.keras.Model: compiled model
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Bidirectional(LSTM(192, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Bidirectional(LSTM(64, return_sequences=False)))

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, name='output'))

    model.compile(optimizer=Adam(lr), loss='mse', metrics=['mae'])
    return model


def build_deeper_small(input_shape, lr=1e-3):
    """Slightly smaller deep model for quick experiments."""
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(lr), loss='mse', metrics=['mae'])
    return model
