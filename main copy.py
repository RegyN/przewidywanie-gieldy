import tensorflow as tf
import numpy as np
import model_utilities as mu
import data_converter as dc
import csv
import keras
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers import Dense, Activation


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def zapisz_historie(dane_hist, sciezka):
    sciezka = ".\logs\\"+sciezka
    plik_csv = open(sciezka, "wt")
    writer_csv = csv.writer(plik_csv, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONE)
    for i, row in enumerate(dane_hist.losses):
        writer_csv.writerow([row])


def wsp_sxy_reg_lin(dane):
    ret = 0
    for i, tick in enumerate(dane):
        ret += (tick[1] * (i + 1))
    return ret


def wsp_sy_reg_lin(dane):
    ret = 0
    for i, tick in enumerate(dane):
        ret += tick[1]
    return ret


def wsp_a_reg_lin(dane):
    n = len(dane)
    return ((n * wsp_sxy_reg_lin(dane)) - ((n * (n + 1) / 2) * wsp_sy_reg_lin(dane))) / (n * n * ((5 * n * n) - 1) / 12)


def zrob_jeden_trening(l_warstw=2, l_kom_ukr=20, bias='true', l_komorek_we=80,
                       l_komorek_wy=80, akt_przejsc='linear', learn_rate=0.15, momentum=0.15, decay=0.0,
                       batch_size=60, l_epok=1, val_split=0.2, trybwartosci=True, typ='lstm'):
    sciezka_csv = ".\gielda.csv"
    tren_input, tren_output = zrob_dane(sciezka_csv, trybwartosci)

    l_wejsc_sieci = 4
    l_wyjsc_sieci = 5

    # Tworzę model sieci
    if typ == 'lstm':
        model = zrob_model_lstm(akt_przejsc, bias, l_kom_ukr, l_komorek_we, l_warstw, l_wejsc_sieci, l_wyjsc_sieci)
    elif typ == 'rnn':
        model = zrob_model_rnn(akt_przejsc, bias, l_kom_ukr, l_komorek_we, l_warstw, l_wejsc_sieci, l_wyjsc_sieci)
    else:
        print("Błędna nazwa typu sieci")
        return
    # Wybieram sposób trenowania sieci
    optimizer = keras.optimizers.SGD(lr=learn_rate, momentum=momentum, decay=decay, nesterov=False)
    model.compile(loss="mean_squared_error", optimizer=optimizer)

    # Trenuję sieć
    history = LossHistory()
    model.fit(tren_input, tren_output, batch_size=batch_size, epochs=l_epok, validation_split=val_split,
              verbose=1, callbacks=[history])
    predicted = model.predict(tren_input)
    print(predicted[15])
    print(tren_output[15])
    print(tren_input[15][-1][1])
    nazwa = ("W"+str(l_warstw)+"K"+str(l_kom_ukr)+"LR"+str(learn_rate)+"M"+str(momentum)+"B"+str(batch_size)
             + "A"+akt_przejsc+".csv")
    zapisz_historie(history, nazwa)


def zrob_model_lstm(akt_przejsc, bias, l_kom_lstm, l_komorek_we, l_warstw, l_wejsc_sieci, l_wyjsc_sieci):
    model = keras.Sequential()
    model.add(Dense(units=l_komorek_we, input_shape=(720, l_wejsc_sieci,), activation=akt_przejsc))
    # Dodaję opcjonalne warstwy LSTM
    for i in range(0, l_warstw - 1):
        model.add(LSTM(units=l_kom_lstm, return_sequences=True, use_bias=bias))
    # Ostatnia warstwa LSTM
    model.add(LSTM(units=l_kom_lstm, return_sequences=False, use_bias=bias))
    # Dodaję warstwy przejściowe, dostosowujące liczbę wyjść
    model.add(Dense(units=l_wyjsc_sieci, activation=akt_przejsc))
    return model


def zrob_model_rnn(akt_przejsc, bias, l_kom_ukr, l_komorek_we, l_warstw, l_wejsc_sieci, l_wyjsc_sieci):
    model = keras.Sequential()
    model.add(Dense(units=l_komorek_we, input_shape=(720, l_wejsc_sieci,), activation=akt_przejsc))
    # Dodaję opcjonalne warstwy LSTM
    for i in range(0, l_warstw - 1):
        model.add(SimpleRNN(units=l_kom_ukr, return_sequences=True, use_bias=bias, activation="tanh"))
    # Ostatnia warstwa LSTM
    model.add(SimpleRNN(units=l_kom_ukr, return_sequences=False, use_bias=bias))
    # Dodaję warstwy przejściowe, dostosowujące liczbę wyjść
    model.add(Dense(units=l_wyjsc_sieci, activation=akt_przejsc))
    return model


def zrob_dane(sciezka_csv, trybwartosci):
    dane = dc.wczytaj_csv(sciezka_csv)
    dane = dc.konwertuj_na_liczby(dane)
    dane = dc.normalizuj_dane(dane)
    dane = dc.dodaj_ruchoma_srednia(dane, 12)
    dane = dc.dodaj_ruchoma_srednia(dane, 30)
    dane_test, dane_tren = dc.przygotuj_dane_tren_i_test(dane, offset=24)
    tren_input = []
    tren_output = []
    dlug_pak = 720
    for i, pakiet in enumerate(dane_tren):
        pakiet_in = pakiet[0:dlug_pak]
        if trybwartosci:
            pakiet_out = [pakiet[dlug_pak + 24 - 1][1],
                          pakiet[dlug_pak + 2 * 24 - 1][1],
                          pakiet[dlug_pak + 3 * 24 - 1][1],
                          pakiet[dlug_pak + 4 * 24 - 1][1],
                          pakiet[dlug_pak + 5 * 24 - 1][1]]
        else:
            a = wsp_a_reg_lin(pakiet)
            trend = 0
            if a > 1 / 20000:
                trend = 5
            elif a > 1 / 200000:
                trend = 4
            elif a > -1 / 200000:
                trend = 3
            elif a > -1 / 20000:
                trend = 2
            else:
                trend = 1
            pakiet_out = trend
        tren_input.append(pakiet_in)
        tren_output.append(pakiet_out)
    return tren_input, tren_output


def main():
    zrob_jeden_trening(l_warstw=2, l_kom_ukr=20, akt_przejsc='linear', l_epok=1, trybwartosci=True,
                       l_komorek_wy=100, l_komorek_we=10, typ='lstm')

    # zrob_jeden_trening(l_warstw=3, l_kom_ukr=32, akt_przejsc='linear', l_epok=5, trybwartosci=True,
    #                    l_komorek_wy=100, l_komorek_we=10, typ='rnn')


main()