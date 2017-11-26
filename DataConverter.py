import csv


def wczytaj_csv(sciezka):
    plik_csv = open(sciezka, "r")
    czytnik_csv = csv.reader(plik_csv, delimiter=',', quotechar='|')

    wynik = []
    for i, row in enumerate(czytnik_csv):
        wynik.append(row)
    return wynik


def konwertuj_na_liczby(dane_wej):
    dane_skonwertowane = []     # tablica zawierająca tablice par floatów
    j = 0
    for i, row in enumerate(dane_wej):
        if len(row) <= 1:
            j += 1
            dane_skonwertowane.append([])
        else:
            temp = [float(row[0]), float(row[1])]
            dane_skonwertowane[j-1].append(temp)
    return dane_skonwertowane


def zrob_falszywe_dane():
    falszywe_dane = []
    for i in range(0, 15):
        nowa_waluta = []
        for j in range(0, 1440):
            nowa_dana = list({1.1, 1.2})
            nowa_waluta.append(nowa_dana)
        falszywe_dane.append(nowa_waluta)
    return falszywe_dane


def przygotuj_dane_tren_i_test_old(dane_wej):
    dlugosc_pakietu = 840   # ile punktow danych ma być w jednej sekwencji dawanej na wejscie LSTM
    liczba_punktow_test = 360   # tyle ostatnich punktow ma nie wchodzic w sklad danych treningowych
    offset = 5              # liczba punktow danych o ktore przesuwam sie tworząc kolejne pakiety
    max_pakietow_test = liczba_punktow_test/offset
    dane_test = []
    dane_tren = []
    for i, waluta in enumerate(dane_wej):   # dziele dane na pojedyncze waluty
        ilosc = len(waluta)
        if ilosc > (dlugosc_pakietu + liczba_punktow_test):

            liczba_pakietow = int((ilosc - dlugosc_pakietu) / offset)
            liczba_pakietow_tren = liczba_pakietow - max_pakietow_test
            for j in range(0, liczba_pakietow):    # tworze pakiety z danej waluty
                volume = []
                value = []
                for k in range(0, dlugosc_pakietu):     # wpisuje punkty do danego pakietu tren.
                    volume.append(dane_wej[i][offset * j][0])
                    value.append(dane_wej[i][offset * j][1])
                pakiet = [volume, value]

                if j <= liczba_pakietow_tren:
                    dane_tren.append(pakiet)
                else:
                    dane_test.append(pakiet)

        print("Zrobiona waluta nr " + str(i))

    return dane_test, dane_tren


def przygotuj_dane_tren_i_test(dane_wej):
    dlugosc_pakietu = 840   # ile punktow danych ma być w jednej sekwencji dawanej na wejscie LSTM
    liczba_punktow_test = 360   # tyle ostatnich punktow ma nie wchodzic w sklad danych treningowych
    offset = 5              # liczba punktow danych o ktore przesuwam sie tworząc kolejne pakiety
    max_pakietow_test = liczba_punktow_test/offset
    dane_test = []
    dane_tren = []
    for i, waluta in enumerate(dane_wej):   # dziele dane na pojedyncze waluty
        ilosc = len(waluta)
        if ilosc > (dlugosc_pakietu + liczba_punktow_test):

            liczba_pakietow = int((ilosc - dlugosc_pakietu) / offset)
            liczba_pakietow_tren = liczba_pakietow - max_pakietow_test
            for j in range(0, liczba_pakietow):    # tworze pakiety z danej waluty
                pakiet = []
                for k in range(0, dlugosc_pakietu):     # wpisuje punkty do danego pakietu tren.
                    pakiet.append([waluta[offset*j+k][0], waluta[offset*j+k][1]])

                if j <= liczba_pakietow_tren:
                    dane_tren.append(pakiet)
                else:
                    dane_test.append(pakiet)

        print("Zrobiona waluta nr " + str(i))

    return dane_test, dane_tren
