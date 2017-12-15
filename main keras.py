import tensorflow as tf
import DataConverter as dc
import csv
import keras
import numpy as np
from keras import optimizers
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def zapisz_historie(dane_hist, sciezka):
    plik_csv = open(sciezka, "wt")
    writer_csv = csv.writer(plik_csv, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONE)
    for i, row in enumerate(dane_hist.losses):
        writer_csv.writerow([row])


<<<<<<< HEAD
def przygotuj_dane_wej():
=======
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


def zrob_jeden_trening(l_warstw=2, l_komorek_lstm=20, bias='true', l_komorek_we=80, l_komorek_wy=80,
                       aktywacja_przejsc='linear', learning_rate=0.15, momentum=0.15, decay=0.0, batch_size=60,
                       l_epok=1, val_split=0.2, offset=24, trybwartosci=True):
>>>>>>> e1c289bcded9f16a8e696f81015b99396f720e67
    sciezka_csv = ".\gielda.csv"
    dane = dc.wczytaj_csv(sciezka_csv)
    dane = dc.konwertuj_na_liczby(dane)
    dane = dc.normalizuj_dane(dane)
    dane = dc.dodaj_ruchoma_srednia(dane, 12)
    dane = dc.dodaj_ruchoma_srednia(dane, 30)
<<<<<<< HEAD
    dane_test, dane_tren = dc.przygotuj_dane_tren_i_test(dane, offset=24)
=======
    dane_test, dane_tren = dc.przygotuj_dane_tren_i_test(dane, offset=offset)

>>>>>>> e1c289bcded9f16a8e696f81015b99396f720e67
    tren_input = []
    tren_output = []
    dlug_pak = 720
    for i, pakiet in enumerate(dane_tren):
        pakiet_in = pakiet[0:dlug_pak]
<<<<<<< HEAD
        pakiet_out = [pakiet[dlug_pak + 24 - 1][1],
                      pakiet[dlug_pak + 2 * 24 - 1][1],
                      pakiet[dlug_pak + 3 * 24 - 1][1],
                      pakiet[dlug_pak + 4 * 24 - 1][1],
                      pakiet[dlug_pak + 5 * 24 - 1][1]]
=======
        if trybwartosci == True:
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
>>>>>>> e1c289bcded9f16a8e696f81015b99396f720e67
        tren_input.append(pakiet_in)
        tren_output.append(pakiet_out)
    return tren_input, tren_output


def zrob_jeden_trening(tren_in, tren_out, l_warstw=2, l_kom_lstm=20, bias='true', l_komorek_we=80,
                       l_komorek_wy=80, akt_przejsc='linear', learn_rate=0.15, momentum=0.15, decay=0.0,
                       batch_size=60, l_epok=1, val_split=0.2):

    l_wejsc_sieci = 4
    l_wyjsc_sieci = 5

    # Tworzę model sekwencyjny
    model = keras.Sequential()
    model.add(Dense(units=l_komorek_we, input_shape=(720, l_wejsc_sieci,), activation=akt_przejsc))

    # Dodaję opcjonalne warstwy LSTM
<<<<<<< HEAD
    for i in range(0, l_warstw-1):
        model.add(LSTM(units=l_kom_lstm, return_sequences=True, use_bias=bias))
=======
    for i in range(0, l_warstw - 1):
        model.add(LSTM(units=l_komorek_lstm, return_sequences=True, use_bias=bias))
>>>>>>> e1c289bcded9f16a8e696f81015b99396f720e67
    # Ostatnia warstwa LSTM
    model.add(LSTM(units=l_kom_lstm, return_sequences=False, use_bias=bias))
    # Dodaję warstwy przejściowe, dostosowujące liczbę wyjść
    model.add(Dense(units=l_komorek_wy, activation=akt_przejsc))
    model.add(Dense(units=l_wyjsc_sieci, activation=akt_przejsc))
    # Wybieram sposób trenowania sieci
    optimizer = keras.optimizers.SGD(lr=learn_rate, momentum=momentum, decay=decay, nesterov=False)
    model.compile(loss="mean_squared_error", optimizer=optimizer)

    # Trenuję sieć
    history = LossHistory()
    model.fit(tren_in, tren_out, batch_size=batch_size, epochs=l_epok, validation_split=val_split,
              verbose=1, callbacks=[history])
<<<<<<< HEAD
    # predicted = model.predict(tren_in)
    # print(predicted[15])
    # print(tren_out[15])
    nazwa = "W"+str(l_warstw)+"K"+str(l_kom_lstm)+"LR"+str(learn_rate)+"M"+str(momentum)+"B"+str(batch_size)\
=======
    predicted = model.predict(tren_input)
    print(predicted[15])
    print(tren_output[15])
    nazwa = "W" + str(l_warstw) + "K" + str(l_komorek_lstm) + "LR" + str(learning_rate) + "M" + str(
        momentum) + "B" + str(batch_size) \
>>>>>>> e1c289bcded9f16a8e696f81015b99396f720e67
            + ".csv"
    zapisz_historie(history, nazwa)


<<<<<<< HEAD
def main() :
    tren_input, tren_output = przygotuj_dane_wej()
    zrob_jeden_trening(tren_in=tren_input, tren_out=tren_output, l_warstw=1, l_epok=2)
    zrob_jeden_trening(tren_in=tren_input, tren_out=tren_output, l_warstw=1, l_epok=2, learn_rate=0.3, momentum=0.3)
    zrob_jeden_trening(tren_in=tren_input, tren_out=tren_output, l_warstw=1, l_epok=2, learn_rate=0.3, momentum=0.1)
    zrob_jeden_trening(tren_in=tren_input, tren_out=tren_output, l_warstw=1, l_epok=4, learn_rate=0.05, momentum=0.05)
    zrob_jeden_trening(tren_in=tren_input, tren_out=tren_output, l_warstw=1, l_kom_lstm=40, l_epok=2)
    zrob_jeden_trening(tren_in=tren_input, tren_out=tren_output, l_warstw=2, l_kom_lstm=15, l_epok=2)
    zrob_jeden_trening(tren_in=tren_input, tren_out=tren_output, l_warstw=2, l_kom_lstm=30, l_epok=2)
=======

def main():
    zrob_jeden_trening(l_warstw=2, l_komorek_lstm=20, offset=25, aktywacja_przejsc='linear', l_epok=1, trybwartosci=True)
>>>>>>> e1c289bcded9f16a8e696f81015b99396f720e67


main()
