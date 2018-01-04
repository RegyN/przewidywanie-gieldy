import tensorflow as tf
import numpy as np
import model_utilities as mu
import data_converter as dc
import csv
import keras
from keras.callbacks import EarlyStopping
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers import Dense, Activation
from wykresy import wczytaj_wykres, rysuj_wykres


class SiecLstmRegresja:
    modelSieci = None

    def __init__(self, l_warstw=2, l_kom_ukr=20, bias='true', l_wejsc=2, f_aktyw='linear', l_wyjsc=5, dl_pak=720):

        self.history = mu.LossHistory()
        self.modelSieci = keras.Sequential()
        self.modelSieci.add(LSTM(input_shape=(dl_pak, l_wejsc,), units=l_kom_ukr, return_sequences=True,
                                 use_bias=bias, activation=f_aktyw))
        for i in range(0, l_warstw - 1):
            self.modelSieci.add(LSTM(units=l_kom_ukr, return_sequences=True, use_bias=bias, activation=f_aktyw))
        self.modelSieci.add(LSTM(units=l_wyjsc, return_sequences=False, use_bias=bias, activation=f_aktyw))
        self.nazwaModelu = ("lstm"+"W"+str(l_warstw)+"K"+str(l_kom_ukr)+"I"+str(l_wejsc)
                            + "O"+str(l_wyjsc)+"A"+f_aktyw)

    def trenuj(self, tren_input, tren_output, learn_rate=0.15, momentum=0.15, decay=0.0,
               batch_size=60, l_epok=1, val_split=0.2):

        optimizer = keras.optimizers.SGD(lr=learn_rate, momentum=momentum, decay=decay, nesterov=False)
        self.modelSieci.compile(loss="mean_squared_error", optimizer=optimizer)
        # Trenuję sieć
        stopper = EarlyStopping(patience=2)
        self.modelSieci.fit(tren_input, tren_output, batch_size=batch_size, epochs=l_epok, validation_split=val_split,
                            verbose=1, callbacks=[self.history, stopper])
        self.kodTreningu = ("LR"+str('%.2F' % learn_rate)+"M"+str('%.2F' % momentum)
                            + "B"+str(batch_size)+"LE"+str(l_epok))

    def zapisz_historie(self):
        sciezka = ".\logs\\" + self.nazwaModelu + self.kodTreningu + "W.csv"
        plik_csv = open(sciezka, "wt")
        writer_csv = csv.writer(plik_csv, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONE)
        for i, row in enumerate(self.history.losses):
            writer_csv.writerow([row])

    def zapisz_wyniki(self):
        mu.zapisz_model(self.modelSieci, self.nazwaModelu + self.kodTreningu + "W.h5")
        self.zapisz_historie()

    def testuj(self, wejscia, wyjscia):
        predicted = self.modelSieci.predict(wejscia)
        for i in range(0, 50):
            print("Rzecz.: "+str(wyjscia[i])+"Przew.: "+str(predicted[i]))
