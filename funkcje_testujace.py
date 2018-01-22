import data_converter as dc
import matplotlib.pyplot as plt
import numpy as np


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


# Wyznaczam błąd procentowy
def wyznacz_sredni_blad(wyj_rzecz, wyj_pred, wejscia):
    blad_suma = 0.0
    for i in range(0, len(wyj_rzecz)):
        blad = (wyj_rzecz[i] - wyj_pred[i])/wyj_rzecz[i]
        if blad >= 0:
            blad_suma = blad_suma + blad
        else:
            blad_suma = blad_suma - blad

    blad = float(blad_suma) / float(len(wyj_rzecz))*100
    return blad


def zrob_testy_stat(siec, typ_sieci='lstm', na_treningowych=False):
    if na_treningowych:
        sciezka = ".\\trening.csv"
    else:
        sciezka = ".\\test.csv"
    testowe = dc.wczytaj_csv(sciezka)
    testowe = dc.konwertuj_na_liczby(testowe)
    liczba_walut = len(testowe)

    trendy = []
    bledy = []

    trendy_calk = 0.0
    trend_max = 0.0
    trend_min = 100.0
    blad_calk = 0.0
    blad_max = 0.0
    blad_min = 100.0


    dlugosc_pakietu = 100
    odleglosc_out = 5
    offset = 1
    for i in range(0, liczba_walut):
        waluta = testowe[i]
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
        if typ_sieci == 'ff':
            testinput = dc.przeksztalc_dane_na_ff(testinput)
        predicted = siec.testuj(testinput)
        for j in range(0, len(predicted)):
            predicted[j] = (predicted[j] * (maxproc - minproc)) + minproc
        predictedoutput = []
        realoutput = []
        ilosc = len(waluta)-1
        if ilosc > dlugosc_pakietu + odleglosc_out:
            liczba_pakietow = int((ilosc - dlugosc_pakietu - odleglosc_out) / offset)
            for j in range(0, len(inputreal)):
                predictedoutput.append((predicted[j] + 1) * wartosci[j])
                realoutput.append((rzeczproc[j] + 1) * wartosci[j])

        zgod = wyznacz_poprawnosc_trendu(realoutput, predictedoutput, wartosci)
        trendy.append(zgod)
        if zgod > trend_max:
            trend_max = zgod
        if zgod < trend_min:
            trend_min = zgod

        trendy_calk = trendy_calk + zgod

        blad = wyznacz_sredni_blad(realoutput, predictedoutput, wartosci)
        bledy.append(blad)
        if blad > blad_max:
            blad_max = blad
        if blad < blad_min:
            blad_min = blad
        blad_calk = blad_calk + blad

    blad_sred = blad_calk/liczba_walut
    zgodnosc_calk = trendy_calk/liczba_walut
    print("Średnia zgodność trendów: "+str(zgodnosc_calk)+"%, a średni błąd procentowy: "+str(blad_sred)+"%")
    print("Max zgodność trendów: "+str(trend_max)+"%, a min: "+str(trend_min)+"%")
    print("Max błąd dla waluty: "+str(blad_max)+"%, a min: "+str(blad_min)+"%")

    plt.hist(bledy, 15, normed=1, facecolor='green', alpha=0.75)
    # plt.plot(x, rzeczproc, 'g-', x, predicted, 'r-')
    plt.grid(True)
    plt.show()

    plt.hist(trendy, 15)
    # plt.plot(x, rzeczproc, 'g-', x, predicted, 'r-')
    plt.grid(True)
    plt.show()

    return zgodnosc_calk, blad_sred


def testuj_trendy(siec, typ_sieci='lstm', na_treningowych=False):
    if na_treningowych:
        sciezka = ".\\trening.csv"
    else:
        sciezka = ".\\test.csv"
    testowe = dc.wczytaj_csv(sciezka)
    testowe = dc.konwertuj_na_liczby(testowe)
    trendy_calk = 0.0
    dlugosc_pakietu = 100
    odleglosc_out = 5
    offset = 1
    for i in range(0, len(testowe)):
        waluta = testowe[i]
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
        if typ_sieci == 'ff':
            testinput = dc.przeksztalc_dane_na_ff(testinput)
        predicted = siec.testuj(testinput)
        for j in range(0, len(predicted)):
            predicted[j] = (predicted[j] * (maxproc - minproc)) + minproc
        predictedoutput = []
        realoutput = []
        ilosc = len(waluta)-1
        if ilosc > dlugosc_pakietu + odleglosc_out:
            liczba_pakietow = int((ilosc - dlugosc_pakietu - odleglosc_out) / offset)
            for j in range(0, len(inputreal)):
                predictedoutput.append((predicted[j] + 1) * wartosci[j])
                realoutput.append((rzeczproc[j] + 1) * wartosci[j])

        zgod = wyznacz_poprawnosc_trendu(realoutput, predictedoutput, wartosci)
        trendy_calk = trendy_calk + zgod
    zgodnosc_calk = trendy_calk/len(testowe)
    return zgodnosc_calk


def testuj_wartosci_proc(siec, typ_sieci='lstm', na_treningowych=False):
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
            if typ_sieci == 'ff':
                testinput = dc.przeksztalc_dane_na_ff(testinput)
            predicted = siec.testuj(testinput)
            for j in range(0, len(predicted)):
                predicted[j] = (predicted[j] * (maxproc - minproc)) + minproc
            predictedoutput = []
            realoutput = []
            ilosc = len(waluta)-1
            if ilosc > dlugosc_pakietu + odleglosc_out:
                liczba_pakietow = int((ilosc - dlugosc_pakietu - odleglosc_out) / offset)
                for j in range(0, len(inputreal)):
                    predictedoutput.append((predicted[j] + 1) * wartosci[j])
                    realoutput.append((rzeczproc[j] + 1) * wartosci[j])

            zgod = wyznacz_poprawnosc_trendu(realoutput, predictedoutput, wartosci)
            print("Poprawnosc trendu to: " + str(zgod) + "%")

            x = np.linspace(0, len(realoutput), len(realoutput))
            plt.plot(x, realoutput, 'g-', x, predictedoutput, 'r-')
            # plt.plot(x, rzeczproc, 'g-', x, predicted, 'r-')
            plt.grid(True)
            plt.show()
        else:
            print("Nie ma takiej waluty, wybierz ponownie")


def wyznacz_poprawnosc_trendu(wyj_rzecz, wyj_pred, wejscia):
    trend_rzecz = []
    trend_pred = []
    for i in range(0, len(wyj_rzecz)):
        poczatek = wejscia[i]
        delta_rzecz = wyj_rzecz[i] - poczatek
        delta_pred = wyj_pred[i] - poczatek
        if delta_rzecz >= 0:
            trend_rzecz.append(1)
        else:
            trend_rzecz.append(0)
        if delta_pred >= 0:
            trend_pred.append(1)
        else:
            trend_pred.append(0)

    zgodne = 0
    for i in range(0, len(trend_rzecz)):
        if trend_rzecz[i] == trend_pred[i]:
            zgodne = zgodne + 1

    poprawnosc = float(zgodne)/float(len(trend_rzecz)) * 100.0
    return poprawnosc
