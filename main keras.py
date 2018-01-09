import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import model_utilities as mu
import data_converter as dc
from siecLstmRegresja import SiecLstmRegresja
from siecLstmKlasyfikacja import SiecLstmKlasyfikacja
import csv
import keras
import copy
from keras.callbacks import EarlyStopping
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers import Dense, Activation
from wykresy import wczytaj_wykres, rysuj_wykres


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def zapisz_historie(dane_hist, sciezka):
    sciezka = ".\logs\\" + sciezka
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


def zrob_jeden_trening(l_warstw=2, l_kom_ukr=20, bias='true', l_komorek_we=20,
                       akt_przejsc='linear', learn_rate=0.15, momentum=0.15, decay=0.0,
                       batch_size=60, l_epok=1, val_split=0.2, trybwartosci=True, typ='lstm'):
    sciezka_csv = ".\gielda.csv"
    tren_input, tren_output = zrob_dane(sciezka_csv, trybwartosci)

    l_wejsc_sieci = 2
    if trybwartosci:
        l_wyjsc_sieci = 5
    else:
        l_wyjsc_sieci = 5

    # Tworzę model sieci
    model = mu.zrob_siec(typ, l_komorek_we, akt_przejsc, bias, l_kom_ukr, l_komorek_we, l_warstw, l_wejsc_sieci,
                         l_wyjsc_sieci)
    if model is None:
        return

    # Wybieram sposób trenowania sieci
    opt = keras.optimizers.RMSprop()
    optimizer = keras.optimizers.SGD(lr=learn_rate, momentum=momentum, decay=decay, nesterov=False)
    model.compile(loss="mean_squared_error", optimizer=optimizer)

    # Trenuję sieć
    history = LossHistory()
    stopper = EarlyStopping(patience=2)
    model.fit(tren_input, tren_output, batch_size=batch_size, epochs=l_epok, validation_split=val_split,
              verbose=1, callbacks=[history, stopper])
    if trybwartosci:
        predicted = model.predict(tren_input)
    else:
        predicted = model.predict_classes(tren_input)
    print(predicted[15])
    print(tren_output[15])
    # print(tren_input[15][-1][1])

    nazwa = (
        typ + "W" + str(l_warstw) + "K" + str(l_kom_ukr) + "LW" + str(l_komorek_we) + "LR" + str('%.2F' % learn_rate)
        + "M" + str('%.2F' % momentum) + "B" + str(batch_size) + "A" + akt_przejsc + "LE" + str(l_epok) + "W")
    if trybwartosci:
        nazwa = nazwa + "W"
    else:
        nazwa = nazwa + "T"
    zapisz_historie(history, nazwa + ".csv")
    mu.zapisz_model(model, nazwa + ".h5")


def zrob_trening(l_warstw=2, l_kom_ukr=20, bias='true', l_komorek_we=20,
                 akt_przejsc='linear', learn_rate=0.15, momentum=0.15, decay=0.0,
                 batch_size=60, l_epok=1, val_split=0.2, trybwartosci=True, typ='lstm', dl_pak=480):
    sciezka_csv = ".\\trening.csv"
    tren_input, tren_output = zrob_dane(sciezka_csv, trybwartosci, dl_pak)

    print("Dane testowe i treningowe gotowe")

    siec = SiecLstmKlasyfikacja(l_warstw, l_kom_ukr, bias, l_wejsc=4, f_aktyw=akt_przejsc, dl_pak=dl_pak)
    siec.trenuj(tren_input, tren_output, learn_rate, momentum, decay, batch_size, l_epok, val_split)
    siec.testuj(tren_input[0:150], tren_output[0:150])
    siec.zapisz_wyniki()


def zrob_dane(sciezka_csv, trybwartosci, dl_pak=480):
    dane = dc.wczytaj_csv(sciezka_csv)
    dane = dc.konwertuj_na_liczby(dane)
    dane = dc.normalizuj_dane(dane)
    dane = np.array(dane)

    dc.dodaj_ruchoma_srednia(dane, 24)
    dc.dodaj_ruchoma_srednia(dane, 60)
    dane_test, dane_tren = dc.przygotuj_dane_tren_i_test(dane, offset=24)
    # dane_test, dane_tren = dc.zrob_dane_eksperymentalne()
    tren_input = []
    tren_output = []
    dlug_pak = dl_pak
    for i, pakiet in enumerate(dane_tren):
        # pakiet_in = pakiet[0:dlug_pak]
        pakiet_in = []
        for j in range(0, dlug_pak):
            pakiet_in.append([pakiet[j][0], pakiet[j][1], pakiet[j][2], pakiet[j][3]])
        if trybwartosci:
            pakiet_out = [pakiet[dlug_pak + 1 * 24 - 1][1],
                          pakiet[dlug_pak + 2 * 24 - 1][1],
                          pakiet[dlug_pak + 3 * 24 - 1][1],
                          pakiet[dlug_pak + 4 * 24 - 1][1],
                          pakiet[dlug_pak + 5 * 24 - 1][1]]
        else:
            a = wsp_a_reg_lin(pakiet[dlug_pak:dlug_pak + 5 * 24])
            if a > 0:  # 1 / 200000:
                pakiet_out = [1, 0]
            else:
                pakiet_out = [0, 1]
        tren_input.append(pakiet_in)
        tren_output.append(pakiet_out)
    return tren_input, tren_output


def testuj_wartosci(siec):
    testowe = dc.wczytaj_csv(".\\test.csv")
    testowe = dc.konwertuj_na_liczby(testowe)
    ilosc_walut = len(testowe)
    print("Wybierz z " + str(ilosc_walut) + " walut numer waluty do przetestowania")
    wybor = ""
    dlugosc_pakietu = 100
    odleglosc_out = 5
    offset = 1
    while True:
        print("")
        wybor = input("Wpisz numer waluty")
        if ilosc_walut > int(wybor) > -1:
            waluta = testowe[int(wybor)]
            procentowo = dc.procentowo(waluta)
            procentowo = np.array(procentowo)
            testinput, rzeczywisteproc = dc.przygotuj_input_output_wartosci([procentowo], offset=offset,
                                                                            sekwencja_danych=dlugosc_pakietu,
                                                                            odleglosc_out=odleglosc_out)
            print("Dane testowe gotowe")
            predicted = siec.testuj(testinput)
            predictedoutput = []
            realoutput = []
            ilosc = len(waluta)
            walutareal = waluta[1:ilosc]
            ilosc = len(walutareal)
            if ilosc > dlugosc_pakietu + odleglosc_out:
                liczba_pakietow = int((ilosc - dlugosc_pakietu - odleglosc_out) / offset)
                for j in range(0, liczba_pakietow):
                    realoutput.append(float(rzeczywisteproc[j][0]) * walutareal[offset * j + dlugosc_pakietu - 1][1])
                    predictedoutput.append(float(predicted[j][0]) * walutareal[offset * j + dlugosc_pakietu - 1][1])
            x = np.linspace(0, len(realoutput), len(realoutput))
            plt.plot(x, realoutput, x, predictedoutput)
            plt.grid(True)
            plt.show()
        else:
            print("Nie ma takiej waluty, wybierz ponownie")


def zrob_trening_wartosci(l_warstw=2, l_kom_ukr=32, bias='true',
                          akt_przejsc='tanh', learn_rate=0.3, momentum=0.3, decay=0.0,
                          batch_size=15, l_epok=3, l_powtorz_tren=10, typ='lstm', dl_pak=100):
    treningowe = dc.wczytaj_csv(".\\trening.csv")
    treningowe = dc.konwertuj_na_liczby(treningowe)
    treningowe = dc.procentowo_dane(treningowe)
    treningowe = np.array(treningowe)
    input, output = dc.przygotuj_input_output_wartosci(treningowe)
    print("Dane treningowe gotowe")
    siec = SiecLstmRegresja(l_warstw=l_warstw, l_kom_ukr=l_kom_ukr, bias=bias, l_wejsc=2, f_aktyw=akt_przejsc,
                            l_wyjsc=1, dl_pak=dl_pak)
    siec.trenuj(input, output, learn_rate, momentum, decay, batch_size, l_epok, l_powtorz_tren)
    siec.zapisz_model()
    return siec


def main():
    print("Przewidywanie gieldy kryptowalut")
    wybor = ""
    while True:
        print("")
        wybor = input("1- uczenie sieci klasyfikacja 2- wczytanie wykresu 3 - uruchom skrypt"
                      " 4 - rysuj wykresy skryptu 5 - uczenie i testowanie regresja 0- wyjscie z programu")
        if int(wybor) == 1:
            zrob_trening(l_warstw=2, l_kom_ukr=20, bias='true', l_komorek_we=32,
                         akt_przejsc='tanh', learn_rate=0.3, momentum=0.3, decay=0.0, dl_pak=240,
                         batch_size=15, l_epok=10, val_split=0.2, trybwartosci=False, typ='lstm')
        elif int(wybor) == 2:
            sciezka = input("Podaj nazwe pliku do wczytania")
            a = wczytaj_wykres(sciezka)
            rysuj_wykres(a)
        elif int(wybor) == 3:
            zrob_jeden_trening(l_warstw=1, l_kom_ukr=32, bias='true', l_komorek_we=32,
                               akt_przejsc='tanh', learn_rate=0.4, momentum=0.8, decay=0.0,
                               batch_size=10, l_epok=20, val_split=0.2, trybwartosci=False, typ='lstm')
            zrob_jeden_trening(l_warstw=3, l_kom_ukr=20, bias='true', l_komorek_we=32,
                               akt_przejsc='tanh', learn_rate=0.4, momentum=0.8, decay=0.0,
                               batch_size=5, l_epok=20, val_split=0.2, trybwartosci=False, typ='lstm')
            if False:
                zrob_jeden_trening(l_warstw=4, l_kom_ukr=20, bias='true', l_komorek_we=32,
                                   akt_przejsc='tanh', learn_rate=0.15, momentum=0.15, decay=0.0,
                                   batch_size=60, l_epok=2, val_split=0.2, trybwartosci=True, typ='lstm')
                zrob_jeden_trening(l_warstw=1, l_kom_ukr=20, bias='true', l_komorek_we=32,
                                   akt_przejsc='tanh', learn_rate=0.15, momentum=0.15, decay=0.0,
                                   batch_size=60, l_epok=2, val_split=0.2, trybwartosci=True, typ='rnn')
                zrob_jeden_trening(l_warstw=1, l_kom_ukr=20, bias='true', l_komorek_we=4,
                                   akt_przejsc='tanh', learn_rate=0.15, momentum=0.15, decay=0.0,
                                   batch_size=60, l_epok=2, val_split=0.2, trybwartosci=True, typ='lstm')
                zrob_jeden_trening(l_warstw=4, l_kom_ukr=20, bias='true', l_komorek_we=32,
                                   akt_przejsc='tanh', learn_rate=0.3, momentum=0.5, decay=0.0,
                                   batch_size=60, l_epok=2, val_split=0.2, trybwartosci=True, typ='lstm')
                zrob_jeden_trening(l_warstw=4, l_kom_ukr=50, bias='true', l_komorek_we=15,
                                   akt_przejsc='tanh', learn_rate=0.15, momentum=0.15, decay=0.0,
                                   batch_size=30, l_epok=1, val_split=0.2, trybwartosci=True, typ='lstm')
                zrob_jeden_trening(l_warstw=15, l_kom_ukr=15, bias='true', l_komorek_we=4,
                                   akt_przejsc='tanh', learn_rate=0.15, momentum=0.3, decay=0.0,
                                   batch_size=60, l_epok=2, val_split=0.2, trybwartosci=True, typ='lstm')
        elif int(wybor) == 4:
            rysuj_wykres(wczytaj_wykres(".\logs\lstmW4K20LW32LR0.15M0.15B60AtanhLE2W.csv"))
            rysuj_wykres(wczytaj_wykres(".\logs\lstmW4K20LW32LR0.15M0.15B60AtanhLE2T.csv"))
            rysuj_wykres(wczytaj_wykres(".\logs\\rnnW1K20LW32LR0.15M0.15B60AtanhLE2W.csv"))
            rysuj_wykres(wczytaj_wykres(".\logs\\rnnW1K20LW32LR0.15M0.15B60AtanhLE2T.csv"))
            rysuj_wykres(wczytaj_wykres(".\logs\lstmW1K20LW4LR0.15M0.15B60AtanhLE2W.csv"))
            rysuj_wykres(wczytaj_wykres(".\logs\lstmW4K20LW32LR0.30M0.50B60AtanhLE2W.csv"))
            rysuj_wykres(wczytaj_wykres(".\logs\lstmW4K50LW15LR0.15M0.15B30AtanhLE1W.csv"))
            rysuj_wykres(wczytaj_wykres(".\logs\lstmW15K15LW4LR0.15M0.30B60AtanhLE2W.csv"))
        elif int(wybor) == 5:
            siec = zrob_trening_wartosci(l_warstw=3, l_kom_ukr=45, bias='true',
                          akt_przejsc='tanh', learn_rate=0.6, momentum=0.7, decay=0.0,
                          batch_size=200, l_epok=1, l_powtorz_tren=2, dl_pak=100)
            #model = mu.wczytaj_model(".\lstmW3K45I2O1AtanhLR0.30M0.60B200LE13W.h5")
            #siec = SiecLstmRegresja()
            #siec.nazwaModelu = "lstmW3K45I2O1Atanh"
            #siec.modelSieci = model
            testuj_wartosci(siec)
        elif int(wybor) == 0:
            break
        else:
            print("Nie ma takiej opcji, wybierz ponownie")


main()
