from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Conv1D, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras import Model


def define_LSTM5(data_in_shape):
    X_input = Input(shape=data_in_shape)
    X = Conv1D(filters=64, kernel_size=5, strides=1, padding='causal', activation='relu')(X_input)
    X = BatchNormalization()(X)
    X = Conv1D(filters=128, kernel_size=3, strides=1, padding='causal', activation='relu')(X)
    X = BatchNormalization()(X) #BatchNormalization layer helps speed up training while making the model more stable.
    #X = Dropout(0.5)(X)

    X = Bidirectional(LSTM(128, return_sequences=True))(X)
    #X = Dropout(0.5)(X) #Dropout layer helps reduce overfitting and makes the model perform better on unseen data
    X = Bidirectional(LSTM(128, return_sequences=True))(X)
    #X = Dropout(0.5)(X)
    X = Bidirectional(LSTM(128, return_sequences=True))(X)
    #X = Dropout(0.5)(X)
    X = Bidirectional(LSTM(128, return_sequences=True))(X)
    #X = Dropout(0.5)(X)
    X = Bidirectional(LSTM(64, return_sequences=False))(X)
    #X = Dropout(0.5)(X)

    X = Dense(512)(X)
    X = LeakyReLU(alpha=0.1)(X) #LeakyReLU is used instead of ReLU in the Dense layer to address potential dead neurons.
    #X = Dropout(0.5)(X)
    X = Dense(256)(X)
    X = LeakyReLU(alpha=0.1)(X)
    #X = Dropout(0.5)(X)
    X = Dense(128)(X)
    X = LeakyReLU(alpha=0.1)(X)

    X_SBP = Dense(1, name='SBP')(X)
    X_DBP = Dense(1, name='DBP')(X)

    model = Model(inputs=X_input, outputs=[X_SBP, X_DBP], name='LSTM5')

    return model