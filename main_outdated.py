import tensorflow as tf
import csv
import data_converter as dc
# import sys
# import os
# import numpy as np
# from random import shuffle


# TODO: Może rozbić main() na parę funkcji, ale DOPIERO jak już będzie działało jak należy.
def main():
    sciezka_csv = ".\gielda.csv"
    dane = dc.wczytaj_csv(sciezka_csv)
    dane = dc.konwertuj_na_liczby(dane)
    dane = dc.normalizuj(dane)
    dane_test, dane_tren = dc.przygotuj_dane_tren_i_test(dane)

    # TODO: Zrobić, żeby było mniej tych zmiennych, bo dużo miejsca w pamięci
    # bo mam dane, dane_test, dane_tren, tren_input i tren_output, a wszystkie w zasadzie zawierają ten sam zestaw
    # danych, tylko w różnych wersjach.

    tren_input = []
    tren_output = []

    for i, pakiet in enumerate(dane_tren):
        pakiet_in = pakiet[0:720]
        pakiet_out = [pakiet[743][1], pakiet[767][1], pakiet[791][1], pakiet[815][1], pakiet[839][1]]
        tren_input.append(pakiet_in)
        tren_output.append(pakiet_out)

    dlugosc_pakietu = 720
    batch_size = 10
    liczba_ukrytych = 24    # Liczba komorek LSTM w warstwie
    l_warstw = 3

    dane_wej = tf.placeholder(tf.float32, [None, dlugosc_pakietu, 2])
    oczekiwane = tf.placeholder(tf.float32, [None, 5])

    weights = tf.Variable(tf.truncated_normal([liczba_ukrytych, int(oczekiwane.get_shape()[1])]))
    bias = tf.Variable(tf.constant(0.1, shape=[oczekiwane.get_shape()[1]]))

    # TODO: Zrobić (jakoś) wiele komórek połączonych warstwami, tak jak opisywaliśmy w konspekcie
    komorka = tf.nn.rnn_cell.LSTMCell(liczba_ukrytych, state_is_tuple=True)
    wyjscia, state = tf.nn.dynamic_rnn(komorka, dane_wej, dtype=tf.float32)

    val = tf.transpose(wyjscia, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)

    prediction = tf.matmul(last, weights) + bias

    loss = tf.reduce_sum(tf.square(prediction - oczekiwane))

    optimizer = tf.train.AdamOptimizer().minimize(loss)

    ###############################################################################################################
    # mistakes = tf.not_equal(oczekiwane, prediction)
    # accuracy = tf.reduce_mean(tf.cast(mistakes, tf.float32))
    # TODO: Zrobić lepszą wersję dokładności, lub pominąć i po prostu oceniać loss
    # Chwilowo źle, wykomentowuje żeby nie spowalniało
    # To jest źle, ale nie wiem jak oceniać dokładność jeśli chodzi o floaty. Powinna mieścić się w zakresie 0 - 1.
    # Im jest większa tym oczywiście lepiej. Poniżej jest wersja dla klasyfikacji używającej kodowania 1 z n,
    # która do naszego projektu się ani trochę nie nadaje, ale chwilowo zostaje, do celów edukacyjnych.
    ###############################################################################################################

    init = tf.global_variables_initializer()

    tf.summary.scalar("loss", loss)
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("biases", bias)
    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        it = 1
        writer = tf.summary.FileWriter('./graphs', graph=tf.get_default_graph())
        batch_num = 0
        sess.run(init)

        while it < 3000:
            #######################################################################################################
            # TODO: Zmienić sposób wyznaczania batchy.
            # W tym momencie próbki podawane są zestawami po kolei tak jak leżą w tablicy. Może warto zmienić na
            # jakiś losowy sposób, ale tak do końca nie wiem.
            batch_x = tren_input[batch_num : batch_num + batch_size]
            batch_y = tren_output[batch_num : batch_num + batch_size]

            _, summary = sess.run([optimizer, summary_op], feed_dict={dane_wej: batch_x, oczekiwane: batch_y})

            if it % 10 == 0:
                writer.add_summary(summary, it)
                # Chwilowo wykomentowałem acc, bo i tak nasza wersja nic nie pokazuje sensownego, po co ma spowalniać.
                # Odkomentować, jak będą poprawione węzły accuracy i mistakes parenaście linijek wyżej
                # acc = sess.run(accuracy, feed_dict={dane_wej: batch_x, oczekiwane: batch_y})
                los = sess.run(loss, feed_dict={dane_wej: batch_x, oczekiwane: batch_y})
                wynik = sess.run(prediction, feed_dict={dane_wej: tren_input[10:11], oczekiwane: tren_output[10:11]})
                print('Uzyskane:    ' + str(wynik))
                print('Rzeczywiste: ' + str(tren_output[10:11]))
                print("Iteracja ", it)
                # print("Accuracy ", acc)
                print("Blad ", los)
                print("__________________")

            it = it + 1
            batch_num += 1
            if batch_num >= len(tren_input) - batch_size - 1:
                batch_num = 0


main()


