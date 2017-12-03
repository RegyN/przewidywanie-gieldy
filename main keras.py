import tensorflow as tf
import DataConverter as dc
import keras
import numpy as np
from keras import optimizers
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation


def main():
    sciezka_csv = ".\gielda.csv"
    dane = dc.wczytaj_csv(sciezka_csv)
    dane = dc.konwertuj_na_liczby(dane)
    dane = dc.normalizuj_dane(dane)
    dane = dc.dodaj_ruchoma_srednia(dane, 5)
    dane = dc.dodaj_ruchoma_srednia(dane, 12)
    dane = dc.dodaj_ruchoma_srednia(dane, 30)
    dane_test, dane_tren = dc.przygotuj_dane_tren_i_test(dane)

    tren_input = []
    tren_output = []

    for i, pakiet in enumerate(dane_tren):
        pakiet_in = pakiet[0:720]
        pakiet_out = [pakiet[743][1], pakiet[767][1], pakiet[791][1], pakiet[815][1], pakiet[839][1]]
        tren_input.append(pakiet_in)
        tren_output.append(pakiet_out)

    dlugosc_pakietu = 720
    batch_size = 25
    l_ukrytych = 50    # Liczba komorek LSTM w warstwie
    l_warstw = 2
    l_wejsc_sieci = 5
    l_wyjsc_sieci = 5

    # Tworzę model sekwencyjny
    model = keras.Sequential()
    model.add(Dense(units=20, input_shape=(720, l_wejsc_sieci,), activation='linear'))
    # Dodaję warstwę LSTM
    model.add(LSTM(units=l_ukrytych, return_sequences=True, use_bias='true'))

    model.add(Dense(units=l_ukrytych, activation='tanh'))
    # Druga warstwa LSTM
    model.add(LSTM(units=l_ukrytych, return_sequences=False, use_bias='true'))
    # Dodaję warstwę przejściową, dostosowującą liczbę wyjść
    model.add(Dense(units=l_wyjsc_sieci, activation='tanh'))
    # Wybieram sposób trenowania sieci
    optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.1, decay=0.0, nesterov=False)
    model.compile(loss="mean_squared_error", optimizer=optimizer)

    # Trenuję sieć
    model.fit(tren_input, tren_output, batch_size=batch_size, epochs=3, validation_split=0.07, verbose=1)
    predicted = model.predict(tren_input)
    # rmse = np.sqrt(((predicted - tren_output) ** 2).mean(axis=0))
    print(predicted[15])
    print(tren_output[15])
    # print(rmse)



main()
