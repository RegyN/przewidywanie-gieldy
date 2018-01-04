import keras
from keras.models import load_model
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers import Dense, Activation


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def zapisz_model(model, sciezka):
    model.save(".\logs\\"+sciezka)


def wczytaj_model(sciezka):
    model = load_model(".\logs\\"+sciezka)
    return model


def wypisz_podsumowanie(model):
    model.summary()


def zrob_siec(typ, l_komorek_we, akt_przejsc, bias, l_kom_ukr, l_kom_lstm, l_warstw, l_wejsc_sieci, l_wyjsc_sieci):
    if typ == 'lstm':
        model = zrob_model_lstm(akt_przejsc, bias, l_kom_ukr, l_komorek_we, l_warstw, l_wejsc_sieci, l_wyjsc_sieci)
    elif typ == 'rnn':
        model = zrob_model_rnn(akt_przejsc, bias, l_kom_ukr, l_komorek_we, l_warstw, l_wejsc_sieci, l_wyjsc_sieci)
    else:
        print("Błędna nazwa typu sieci")
        model = None
    return model


def zrob_model_lstm(akt_przejsc, bias, l_kom_ukr, l_kom_lstm, l_warstw, l_wejsc_sieci, l_wyjsc_sieci):
    model = keras.Sequential()
    # model.add(Dense(units=l_komorek_we,  activation=akt_przejsc))
    model.add(LSTM(input_shape=(720, l_wejsc_sieci,), units=l_kom_lstm, return_sequences=True, use_bias=bias,
                   activation=akt_przejsc))
    for i in range(0, l_warstw - 1):
        model.add(LSTM(units=l_kom_ukr, return_sequences=True, use_bias=bias, activation=akt_przejsc))
    # Dodaję warstwy przejściowe, dostosowujące liczbę wyjść
    model.add(LSTM(units=l_wyjsc_sieci, return_sequences=False, use_bias=bias, activation=akt_przejsc))
    # model.add(Dense(units=l_wyjsc_sieci, activation=akt_przejsc))
    return model


def zrob_model_rnn(akt_przejsc, bias, l_kom_ukr, l_komorek_we, l_warstw, l_wejsc_sieci, l_wyjsc_sieci):
    model = keras.Sequential()
    model.add(Dense(units=l_komorek_we, input_shape=(720, l_wejsc_sieci,), activation=akt_przejsc))
    # Dodaję opcjonalne warstwy LSTM
    for i in range(0, l_warstw - 1):
        model.add(SimpleRNN(units=l_kom_ukr, return_sequences=True, use_bias=bias, activation=akt_przejsc))
    # Ostatnia warstwa LSTM
    model.add(SimpleRNN(units=l_kom_ukr, return_sequences=False, use_bias=bias))
    # Dodaję warstwy przejściowe, dostosowujące liczbę wyjść
    model.add(Dense(units=l_wyjsc_sieci, activation=akt_przejsc))
    return model

