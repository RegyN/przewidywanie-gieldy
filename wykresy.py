import matplotlib.pyplot as plt
import numpy as np
import csv


def wczytaj_wykres(sciezka):
    plik_csv = open(sciezka, "r")
    czytnik_csv = csv.reader(plik_csv, delimiter=',', quotechar='|')

    wynik = []
    for i, row in enumerate(czytnik_csv):
        if len(row) == 1:
            wynik.append(float(row[0]))
    return wynik


def rysuj_wykres(dane):
    x = np.linspace(0, len(dane), len(dane))
    plt.plot(x, dane)
    plt.grid(True)
    plt.show()


