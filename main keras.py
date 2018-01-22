import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import model_utilities as mu
import data_converter as dc
import funkcje_testujace as ft
from siecLstmRegresja import SiecLstmRegresja
from siecFFRegresja import SiecFFRegresja
from siecLstmKlasyfikacja import SiecLstmKlasyfikacja
import csv
import keras
import copy
from wykresy import wczytaj_wykres, rysuj_wykres


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def zapisz_historie(dane_hist, sciezka):
    sciezka = ".\\files\\" + sciezka
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
    sciezka_csv = ".\\files\\trening.csv"
    tren_input, tren_output = zrob_dane(sciezka_csv, trybwartosci, dl_pak)

    print("Dane testowe i treningowe gotowe")
    if typ == 'lstm':
        siec = SiecLstmRegresja(l_warstw, l_kom_ukr, bias, l_wejsc=2, f_aktyw=akt_przejsc, dl_pak=dl_pak)
    elif typ == 'ff':
        siec = SiecFFRegresja(l_warstw, l_kom_ukr, bias, l_wejsc=2, f_aktyw=akt_przejsc, dl_pak=dl_pak)
    else:
        print("Błąd: Wybrano nieprawidłowy typ sieci")
        return None
    print("Sieć gotowa, rozpoczynam trening")
    siec.trenuj(tren_input, tren_output, learn_rate, momentum, decay, batch_size, l_epok, val_split)
    print("Trening zakończony, zapisuję dziennik")
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


def zrob_trening_wartosci(dane, l_warstw=2, l_kom_ukr=32, bias='true',
                          akt_przejsc='tanh', learn_rate=0.3, momentum=0.3, decay=0.0,
                          batch_size=15, l_epok=3, l_powtorz_tren=10, typ='lstm'):
    treningowe = dc.wczytaj_csv(".\\files\\"+dane)
    treningowe = dc.konwertuj_na_liczby(treningowe)
    treningowe = dc.procentowo_dane(treningowe)
    treningowe = np.array(treningowe)
    input, output, min, max = dc.przygotuj_input_output_wartosci(treningowe)
    print("Dane treningowe gotowe")

    if typ == 'lstm':
        siec = SiecLstmRegresja(l_warstw, l_kom_ukr, bias, l_wejsc=2, f_aktyw=akt_przejsc, dl_pak=100)
    elif typ == 'ff':
        siec = SiecFFRegresja(l_warstw, l_kom_ukr, bias, l_wejsc=2, f_aktyw=akt_przejsc, dl_pak=100)
    else:
        print("Błąd: Wybrano nieprawidłowy typ sieci")
        return None

    siec.trenuj(input, output, learn_rate, momentum, decay, batch_size, l_epok, l_powtorz_tren)
    siec.zapisz_model()
    return siec


def main():
    print("Przewidywanie gieldy kryptowalut")
    wybor = ""
    while True:
        print("")
        wybor = input("1- trenuj sieć\n2 - testy statystyczne\n3 - wykresy predykcji\n0- wyjscie z programu\n")
        if int(wybor) == 1:
            l_warstw = input("Podaj liczbę warstw sieci: ")
            l_kom_ukr = input("Podaj liczbę komórek ukrytych sieci: ")
            typ_sieci = input("Podaj typ sieci (lstm lub ff): ")
            akt_przejsc = input("Podaj funkcję aktywacji przejścia(sigmoid/tanh/relu/softsign/softplus/linear): ")
            learn_rate = input("Podaj szybkość nauki: ")
            momentum = input("Podaj pęd nauki: ")
            decay = input("Podaj wskaźnik zanikania nauki: ")
            batch_size = input("Podaj długość batcha: ")
            l_epok = input("Podaj liczbę epok: ")
            l_powtorz_tren = input("Podaj liczbę powtórzeń treningu: ")
            dane = input("Podaj nazwę pliku z danymi treningowymi: ")
            zrob_trening_wartosci(dane, l_warstw=int(l_warstw), l_kom_ukr=int(l_kom_ukr), bias='true', typ=typ_sieci,
                                  akt_przejsc=akt_przejsc, learn_rate=float(learn_rate), momentum=float(momentum),
                                  decay=float(decay),
                                  batch_size=int(batch_size), l_epok=int(l_epok), l_powtorz_tren=int(l_powtorz_tren))
        elif int(wybor) == 2:
            sciezka = input("Podaj nazwe pliku do wczytania: ")
            dane = input("Podaj nazwę pliku z danymi testowymi: ")
            model = mu.wczytaj_model(sciezka)
            if sciezka.startswith("ff"):
                siec = SiecFFRegresja()
            else:
                siec = SiecLstmRegresja()
            siec.modelSieci = model
            ft.zrob_testy_stat(siec, dane)
        elif int(wybor) == 3:
            sciezka = input("Podaj nazwe pliku do wczytania: ")
            dane = input("Podaj nazwę pliku z danymi testowymi: ")
            model = mu.wczytaj_model(sciezka)
            if sciezka.startswith("ff"):
                siec = SiecFFRegresja()
                typ_sieci = "ff"
            else:
                siec = SiecLstmRegresja()
                typ_sieci = "lstm"
            siec.modelSieci = model
            ft.testuj_wartosci_proc(siec, dane, typ_sieci)
        elif int(wybor) == 0:
            break
        else:
            print("Nie ma takiej opcji, wybierz ponownie")


main()
