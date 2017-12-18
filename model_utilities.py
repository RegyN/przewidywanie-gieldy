# Jakieś funkcje działające na modelach, typu zapisywanie i wczytywanie
# TODO: Sprawdzić czy te funkcje w ogóle działają, na szybko zanotowałem z tutoriala
# TODO: Zrobić sprawdzanie czy coś się wczytało/zapisało i ewentualne zwracanie błędów


import keras
from keras.models import load_model


def zapisz_model(model, sciezka):
    model.save(".\logs\\"+sciezka)


def wczytaj_model(sciezka):
    model = load_model(".\logs\\"+sciezka)
    return model


def wypisz_podsumowanie(model):
    model.summary()
