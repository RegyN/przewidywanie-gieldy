import tensorflow as tf
import csv
import DataConverter as dc
# import sys
# import os
# import numpy as np
# from random import shuffle


def true_main():
    sciezka_csv = "D:\Pobrane\gielda.csv"
    dane = dc.wczytaj_csv(sciezka_csv)
    dane_po_konwersji = dc.konwertuj_na_liczby(dane)
    print(len(dane_po_konwersji))
    for i, row in enumerate(dane_po_konwersji):
        print(len(row))


def main():
    sciezka_csv = "D:\Pobrane\gielda.csv"
    dane = dc.wczytaj_csv(sciezka_csv)
    dane = dc.konwertuj_na_liczby(dane)
    dane_test, dane_tren = dc.przygotuj_dane_tren_i_test(dane)

    tren_input = []
    tren_output = []

    for i, pakiet in enumerate(dane_tren):
        pakiet_in = pakiet[0:720]
        pakiet_out = [pakiet[743][1], pakiet[767][1], pakiet[791][1], pakiet[815][1], pakiet[839][1]]
        tren_input.append(pakiet_in)
        tren_output.append(pakiet_out)

    dlugosc_pakietu = 720
    batch_size = 10
    liczba_ukrytych = 24

    dane_wej = tf.placeholder(tf.float32, [None, dlugosc_pakietu, 2])
    oczekiwane = tf.placeholder(tf.float32, [None, 5])

    weights = tf.Variable(tf.truncated_normal([liczba_ukrytych, int(oczekiwane.get_shape()[1])]))
    bias = tf.Variable(tf.constant(0.1, shape=[oczekiwane.get_shape()[1]]))

    komorka = tf.nn.rnn_cell.LSTMCell(liczba_ukrytych, state_is_tuple=True)
    wyjscia, state = tf.nn.dynamic_rnn(komorka, dane_wej, dtype=tf.float32)

    val = tf.transpose(wyjscia, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)

    prediction = tf.matmul(last, weights) + bias

    cross_entropy = -tf.reduce_sum(tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))

    optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

    mistakes = tf.not_equal(tf.argmax(oczekiwane, axis=1), tf.argmax(prediction, axis=1))
    accuracy = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    init = tf.global_variables_initializer()

    tf.summary.scalar("accuracy", accuracy)
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("biases", bias)
    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        it = 1
        writer = tf.summary.FileWriter('./graphs', graph=tf.get_default_graph())
        batch_num = 0
        sess.run(init)

        while it < 100:
            batch_x = tren_input[batch_num*batch_size:batch_num*batch_size+batch_size]
            batch_y = tren_output[batch_num * batch_size:batch_num * batch_size + batch_size]

            _, summary = sess.run([optimizer, summary_op], feed_dict={dane_wej: batch_x, oczekiwane: batch_y})
            writer.add_summary(summary, it)

            if it % 10 == 0:
                acc = sess.run(accuracy, feed_dict={dane_wej: batch_x, oczekiwane: batch_y})
                # los = sess.run(loss, feed_dict={dane_wej: batch_x, oczekiwane: batch_y})
                print("For iter ", it)
                print("Accuracy ", acc)
                # print("Loss ", los)
                print("__________________")

            it = it + 1
            batch_num += 1


main()


