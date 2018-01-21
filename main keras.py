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


def zrob_trening(l_warstw=2, l_kom_ukr=20, bias='true', l_komorek_we=20,
                 akt_przejsc='linear', learn_rate=0.15, momentum=0.15, decay=0.0,
                 batch_size=60, l_epok=1, val_split=0.2, trybwartosci=True, typ='lstm', dl_pak=480):
    sciezka_csv = ".\\trening.csv"
    tren_input, tren_output = zrob_dane(sciezka_csv, trybwartosci, dl_pak)

    print("Dane testowe i treningowe gotowe")

    siec = SiecLstmRegresja(l_warstw, l_kom_ukr, bias, l_wejsc=2, f_aktyw=akt_przejsc, dl_pak=dl_pak)
    siec.trenuj(tren_input, tren_output, learn_rate, momentum, decay, batch_size, l_epok, val_split)
    siec.zapisz_wyniki()

    return siec


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
            pakiet_in.append([pakiet[j][0], pakiet[j][1]])
        if trybwartosci:
            pakiet_out = [pakiet[dlug_pak + 1 * 24 - 1][1]]
        else:
            a = wsp_a_reg_lin(pakiet[dlug_pak:dlug_pak + 5 * 24])
            if a > 0:  # 1 / 200000:
                pakiet_out = [1, 0]
            else:
                pakiet_out = [0, 1]
        tren_input.append(pakiet_in)
        tren_output.append(pakiet_out)
    return tren_input, tren_output


def testuj_wartosci_norm(siec):
    testowe = dc.wczytaj_csv(".\\test.csv")
    testowe = dc.konwertuj_na_liczby(testowe)
    testowe = dc.normalizuj_dane(testowe)
    ilosc_walut = len(testowe)
    print("Wybierz z " + str(ilosc_walut) + " walut numer waluty do przetestowania")
    wybor = ""
    dlugosc_pakietu = 100
    odleglosc_out = 24
    offset = 1
    while True:
        print("")
        wybor = input("Wpisz numer waluty")
        if ilosc_walut > int(wybor) > -1:
            waluta = testowe[int(wybor)]
            waluta = np.array(waluta)

            pred = []
            real = []
            for i, row in enumerate(waluta):
                if i >= len(waluta) - dlugosc_pakietu - odleglosc_out:
                    break
                pred.append(waluta[i:i + 100])
                real.append(waluta[i + 100 + odleglosc_out][1])
            pred = np.array(pred)
            predicted = siec.testuj(pred)
            x = np.linspace(0, len(real), len(real))
            plt.plot(x, real, "g-", x, predicted, "r-")
            plt.grid(True)
            plt.show()
        else:
            print("Nie ma takiej waluty, wybierz ponownie")


def testuj_wartosci_proc(siec, na_treningowych=False):
    if na_treningowych:
        sciezka = ".\\trening.csv"
    else:
        sciezka = ".\\test.csv"
    testowe = dc.wczytaj_csv(sciezka)
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
            inputreal, tmp, min, max = dc.przygotuj_input_output_wartosci([waluta], offset=offset,
                                                                          sekwencja_danych=dlugosc_pakietu,
                                                                          odleglosc_out=odleglosc_out)
            testinput, rzeczywisteproc, minproc, maxproc = dc.przygotuj_input_output_wartosci([procentowo],
                                                                                              offset=offset,
                                                                                              sekwencja_danych=dlugosc_pakietu,
                                                                                              odleglosc_out=odleglosc_out)
            rzeczproc = []
            wartosci = []
            for j in range(0, len(inputreal)):
                wartosci.append(inputreal[j][dlugosc_pakietu - 1][1] * (max - min) + min)
            for j in range(0, len(rzeczywisteproc)):
                rzeczproc.append((rzeczywisteproc[j][0] * (maxproc - minproc)) + minproc)
            print("Dane testowe gotowe")
            predicted = siec.testuj(testinput)
            for j in range(0, len(predicted)):
                predicted[j] = (predicted[j] * (maxproc - minproc)) + minproc
            predictedoutput = []
            realoutput = []
            ilosc = len(waluta)
            walutareal = waluta[1:ilosc]
            ilosc = len(walutareal)
            if ilosc > dlugosc_pakietu + odleglosc_out:
                liczba_pakietow = int((ilosc - dlugosc_pakietu - odleglosc_out) / offset)
                for j in range(0, len(inputreal)):
                    predictedoutput.append((predicted[j] + 1) * wartosci[j])
                    realoutput.append((rzeczproc[j] + 1) * wartosci[j])
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
    input, output, min, max = dc.przygotuj_input_output_wartosci(treningowe)
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
        wybor = input("1- ucz siec 2- wczytaj wykres loss 3 - uruchom skrypt"
                      " 4 - rysuj wykresy 5 - testuj regresje 0- wyjscie z programu")
        if int(wybor) == 1:
            siec = zrob_trening(l_warstw=8, l_kom_ukr=20, bias='true', l_komorek_we=32,
                         akt_przejsc='tanh', learn_rate=0.8, momentum=0.8, decay=0.0, dl_pak=100,
                         batch_size=60, l_epok=50, val_split=0.2, trybwartosci=True, typ='lstm')
            testuj_wartosci_norm(siec)
        elif int(wybor) == 2:
            sciezka = input("Podaj nazwe pliku do wczytania")
            a = wczytaj_wykres(sciezka)
            rysuj_wykres(a)
        elif int(wybor) == 3:
            print("Wybrana funkcja czasowo niedostępna")
        elif int(wybor) == 4:
            print("Wybrana funkcja czasowo niedostępna")
        elif int(wybor) == 5:
            siec = zrob_trening_wartosci(l_warstw=3, l_kom_ukr=45, bias='true',
                         akt_przejsc='tanh', learn_rate=0.6, momentum=0.7, decay=0.0,
                         batch_size=60, l_epok=20, l_powtorz_tren=2, dl_pak=100)
            # model = mu.wczytaj_model("lstmW3K45I2O1AtanhLR0.20M0.70B200LE20W.h5")
            # siec = SiecLstmRegresja()
            # siec.nazwaModelu = "lstmW2K20I4O5Atanh"
            # siec.modelSieci = model
            testuj_wartosci_proc(siec)
        elif int(wybor) == 0:
            break
        else:
            print("Nie ma takiej opcji, wybierz ponownie")


main()
