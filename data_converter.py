import csv
import random as rd


def wczytaj_csv(sciezka):
    plik_csv = open(sciezka, "r")
    czytnik_csv = csv.reader(plik_csv, delimiter=',', quotechar='|')

    wynik = []
    for i, row in enumerate(czytnik_csv):
        wynik.append(row)
    return wynik


# TODO: Może połączyć z wczytaj_csv, ale bez pośpiechu
# Po wczytaniu .csv dane są, a przynajmniej wcześniej były, w formie string.
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


def normalizuj(waluta):
    max = 0
    for i, punkt in enumerate(waluta):
        if punkt[1] > max:
            max = punkt[1]
    for i, punkt in enumerate(waluta):
        waluta[i][1] = punkt[1]/max

    return waluta


def normalizuj_dane(dane_wej):
    for i, waluta in enumerate(dane_wej):
        dane_wej[i] = normalizuj(waluta)

    return dane_wej


# Dzielę cały zestaw danych wejściowych na testowe i treningowe. Cała metodyka pewnie do poprawy, ale z grubsza robi
# co trzeba. Jak zmieni się offset to będzie dużo więcej danych, ale mniej się od siebie nawzajem różniących
def przygotuj_dane_tren_i_test(dane_wej, offset=24, dlug_pak=840, l_pkt_test=360):
    dlugosc_pakietu = dlug_pak      # ile punktow danych ma być w jednej sekwencji dawanej na LSTM ( (30d.+5d.) x 24h)
    liczba_punktow_test = l_pkt_test       # tyle ostatnich punktow ma nie wchodzic w sklad danych treningowych
    max_pakietow_test = liczba_punktow_test/offset
    dane_test = []
    dane_tren = []
    for i, waluta in enumerate(dane_wej):   # dziele dane na pojedyncze waluty
        ilosc = len(waluta)
        if ilosc > (dlugosc_pakietu + liczba_punktow_test):     # Jeśli jest z czego tworzyć dane

            liczba_pakietow = int((ilosc - dlugosc_pakietu) / offset)
            liczba_pakietow_tren = liczba_pakietow - max_pakietow_test
            for j in range(0, liczba_pakietow):    # tworze pakiety z danej waluty
                pakiet = []
                for k in range(0, dlugosc_pakietu):     # wpisuje punkty do danego pakietu tren.
                    pakiet.append([waluta[offset*j+k][0],
                                   waluta[offset*j+k][1],
                                   waluta[offset*j+k][2]])

                if j <= liczba_pakietow_tren:
                    dane_tren.append(pakiet)
                else:
                    dane_test.append(pakiet)
    print("Dane testowe i treningowe gotowe")
    return dane_test, dane_tren


def zrob_dane_eksperymentalne(dlug_pak=840, liczba_pak=1500):
    rd.seed(123)
    dane_test = []
    dane_tren = []
    for i in range(0, liczba_pak):    # tworze pakiety z danej waluty
        pakiet = []
        for j in range(0, dlug_pak):     # wpisuje punkty do danego pakietu tren.
            pakiet.append([0, ((j+1) + rd.random()/3 - 0.16)/840])

        dane_tren.append(pakiet)
    print("Eksperymentalne dane testowe i treningowe gotowe")
    return dane_test, dane_tren


def dodaj_ruchoma_srednia(dane_wej, dlugosc):

    skladniki = []
    for i, waluta in enumerate(dane_wej):
        suma = 0.0
        for j, row in enumerate(waluta):
            if j < dlugosc:
                suma = suma + row[1]
                skladniki.append(row[1])
                row.append(suma/(j+1))
            else:
                suma = suma + row[1] - skladniki.pop(0)
                skladniki.append(row[1])
                row.append(suma/dlugosc)

    return dane_wej




