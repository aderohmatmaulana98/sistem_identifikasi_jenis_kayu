from flask import Flask, render_template, app, redirect, url_for, request
import os
import cv2
import numpy as np
import pandas as pd
import math
import imutils.paths as paths
from flask_mysqldb import MySQL, MySQLdb
import sqlalchemy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from collections import Counter
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["SECRET_KEY"] = "rahasia"
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'db_kayu'
ALLOWED_EXTENSION = set(['png', 'jpeg', 'jpg'])
app.config['UPLOAD_FOLDER'] = 'uploads'
mysql = MySQL(app)
cnx = sqlalchemy.create_engine('mysql+pymysql://root:@localhost:3306/db_kayu')

knn_jarak = []
knn_jarak1 = []


class KNN:
    def __init__(self, k=3):
        self.K = k

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_prediksi = [self._prediksi(x) for x in X]
        return np.array(y_prediksi)

    def _prediksi(self, x):
        # Hitung Jarak ke semua data training
        jarak_titik = [self.jarak(x, x_train) for x_train in self.X_train]
        print(jarak_titik)
        if knn_jarak == []:
            knn_jarak.append(jarak_titik)
        else:
            knn_jarak[0:] = [jarak_titik]
        # urutkan berdasarkan jarak terdekat, ambil sejumlah K
        k_terbaik = np.argsort(jarak_titik)[0:int(self.K)]
        print(k_terbaik)
        # Ambil label k_terbaik
        label_k_terbaik = [self.y_train[i] for i in k_terbaik]
        # voting yang paling banyak
        hasil_voting = Counter(label_k_terbaik).most_common(1)
        return hasil_voting[0][0]

    def jarak(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))


@app.route("/")
def main():
    cur = mysql.connection.cursor()
    cur.execute("SELECT count(*) FROM ekstraksi_fitur")
    total_data = np.array(cur.fetchall())
    total_data = int(total_data)

    cur.execute("SELECT count(*) FROM uji")
    uji = np.array(cur.fetchall())
    uji = int(uji)

    cur.execute("SELECT count(*) FROM latih")
    latih = np.array(cur.fetchall())
    latih = int(latih)
    cur.close()
    return render_template("index.html", menu='dashboard', total_data=total_data, uji=uji, latih=latih)


@app.route("/dataset")
def dataset():
    cur = mysql.connection.cursor()
    cur.execute("SELECT*FROM ekstraksi_fitur")
    ekstraksi = cur.fetchall()
    cur.close()
    cur1 = mysql.connection.cursor()
    cur1.execute("SELECT count(*) FROM ekstraksi_fitur")
    cek_data = np.array(cur1.fetchall())
    print(int(cek_data[0]))
    return render_template('dataset.html', data=ekstraksi, cek_data=cek_data)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION


@app.route("/ekstraksi_fitur")
def ekstraksi_fitur():
    path = r"D:/Kampus/SMT8/belajar/sistem-identifikasi-kayu/dataset/"

    for file in os.listdir(path):
        image = cv2.imread(os.path.abspath(path+"/"+file))
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resize_image = cv2.resize(img, (100, 100))

        derajat_0 = derajat0(resize_image)
        derajat_45 = derajat45(resize_image)
        derajat_90 = derajat90(resize_image)
        derajat_135 = derajat135(resize_image)

        energi_0 = energi(derajat_0)
        homogenity_0 = homogenitas(derajat_0)
        entropy_0 = entropy(derajat_0)
        contras_0 = contras(derajat_0)
        energi_45 = energi(derajat_45)
        homogenity_45 = homogenitas(derajat_45)
        entropy_45 = entropy(derajat_45)
        contras_45 = contras(derajat_45)
        energi_90 = energi(derajat_90)
        homogenity_90 = homogenitas(derajat_90)
        entropy_90 = entropy(derajat_90)
        contras_90 = contras(derajat_90)
        energi_135 = energi(derajat_135)
        homogenity_135 = homogenitas(derajat_135)
        entropy_135 = entropy(derajat_135)
        contras_135 = contras(derajat_135)

        label = 0
        if file.startswith("jati"):
            label = 1
        else:
            label = 2

        fitur = [file, energi_0, homogenity_0, entropy_0, contras_0,
                 energi_45, homogenity_45, entropy_45, contras_45,
                 energi_90, homogenity_90, entropy_90, contras_90,
                 energi_135, homogenity_135, entropy_135, contras_135,
                 label]

        cur = mysql.connection.cursor()
        sql = "INSERT INTO ekstraksi_fitur (file,energy_0,homogenity_0, entropy_0, contras_0,energy_45,homogenity_45, entropy_45, contras_45,energy_90,homogenity_90, entropy_90, contras_90,energy_135,homogenity_135, entropy_135, contras_135, label) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        val = (fitur[0], round(fitur[1], 5), round(fitur[2], 5), round(fitur[3], 5), round(fitur[4], 5), round(fitur[5], 5), round(fitur[6], 5), round(fitur[7], 5), round(fitur[8], 5),
               round(fitur[9], 5), round(fitur[10], 5), round(fitur[11], 5), round(fitur[12], 5), round(fitur[13], 5), round(fitur[14], 5), round(fitur[15], 5), round(fitur[16], 5), fitur[17])
        cur.execute(sql, val)
        mysql.connection.commit()
    return redirect(url_for('dataset'))


def derajat0(img):
    max = np.max(img)
    imgTmp = np.zeros([max+1, max+1])
    for i in range(len(img)):
        for j in range(len(img[i])-1):
            imgTmp[img[i, j], img[i, j+1]] += 1

    transpos = np.transpose(imgTmp)
    data = imgTmp+transpos

    tmp = 0
    for i in range(len(data)):
        for j in range(len(data)):
            tmp += data[i, j]

    for i in range(len(data)):
        for j in range(len(data)):
            data[i, j] /= tmp
    return data


def derajat45(img):
    max = np.max(img)
    imgTmp = np.zeros([max+1, max+1])
    for i in range(len(img)-1):
        for j in range(len(img[i])-1):
            imgTmp[img[i+1, j], img[i, j+1]] += 1

    transpos = np.transpose(imgTmp)
    data = imgTmp+transpos

    tmp = 0
    for i in range(len(data)):
        for j in range(len(data)):
            tmp += data[i, j]

    for i in range(len(data)):
        for j in range(len(data)):
            data[i, j] /= tmp
    return data


def derajat90(img):
    max = np.max(img)
    imgTmp = np.zeros([max+1, max+1])
    for i in range(len(img)-1):
        for j in range(len(img[i])):
            imgTmp[img[i+1, j], img[i, j]] += 1

    transpos = np.transpose(imgTmp)
    data = imgTmp+transpos

    tmp = 0
    for i in range(len(data)):
        for j in range(len(data)):
            tmp += data[i, j]

    for i in range(len(data)):
        for j in range(len(data)):
            data[i, j] /= tmp
    return data


def derajat135(img):
    max = np.max(img)
    imgTmp = np.zeros([max+1, max+1])
    for i in range(len(img)-1):
        for j in range(len(img[i])-1):
            imgTmp[img[i, j], img[i+1, j+1]] += 1

    transpos = np.transpose(imgTmp)
    data = imgTmp+transpos

    tmp = 135
    for i in range(len(data)):
        for j in range(len(data)):
            tmp += data[i, j]

    for i in range(len(data)):
        for j in range(len(data)):
            data[i, j] /= tmp
    return data


def contras(data):
    contras = 0
    for i in range(len(data)):
        for j in range(len(data)):
            contras += data[i, j]*pow(i-j, 2)
    return contras


def entropy(data):
    entro = 0
    for i in range(len(data)):
        for j in range(len(data)):
            if data[i, j] > 0.0:
                entro += -(data[i, j] * math.log(data[i, j]))
    return entro


def homogenitas(data):
    homogen = 0
    for i in range(len(data)):
        for j in range(len(data)):
            homogen += data[i, j]*(1+(pow(i-j, 2)))
    return homogen


def energi(data):
    ener = 0
    for i in range(len(data)):
        for j in range(len(data)):
            ener += data[i, j]**2
    return ener


@app.route("/data_latih")
def data_latih():
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM latih")
    q_hasil = cursor.fetchall()
    cursor.close
    return render_template('data_latih.html', data=q_hasil)


def datalatih():
    df = pd.read_sql_query("SELECT * FROM ekstraksi_fitur", cnx)
    x = df[['file', 'energy_0', 'homogenity_0', 'entropy_0', 'contras_0', 'energy_45', 'homogenity_45', 'entropy_45', 'contras_45',
            'energy_90', 'homogenity_90', 'entropy_90', 'contras_90', 'energy_135', 'homogenity_135', 'entropy_135', 'contras_135', 'label']].values
    Z_train,  W_train = train_test_split(x, test_size=0.2, random_state=42)
    latih = Z_train
    return latih


@app.route("/import_data_latih")
def import_data_latih():
    latih = datalatih()
    print(latih[0])
    for citra in latih:
        files = r"D:/Kampus/SMT8/belajar/sistem-identifikasi-kayu/dataset/" + \
            citra[0]
        gambar = cv2.imread(os.path.abspath(files))
        hasil = r"D:/Kampus/SMT8/belajar/sistem-identifikasi-kayu/uploads/data_latih/" + \
            citra[0]
        cv2.imwrite(hasil, gambar)
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT count(*) FROM latih")
    hasil = np.array(cursor.fetchall())
    jum_data_latih = int(hasil[0][0])
    if jum_data_latih == 0:
        for d_latih in latih:
            cur = mysql.connection.cursor()
            sql = "INSERT INTO latih (file,energy_0,homogenity_0, entropy_0, contras_0,energy_45,homogenity_45, entropy_45, contras_45,energy_90,homogenity_90, entropy_90, contras_90,energy_135,homogenity_135, entropy_135, contras_135, label) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            val = (d_latih[0], d_latih[1], d_latih[2], d_latih[3], d_latih[4],
                   d_latih[5], d_latih[6], d_latih[7], d_latih[8], d_latih[9], d_latih[10], d_latih[11], d_latih[12], d_latih[13], d_latih[14], d_latih[15], d_latih[16], d_latih[17])
            cur.execute(sql, val)
            mysql.connection.commit()
    return redirect(url_for('data_latih'))


def datauji():
    df = pd.read_sql_query("SELECT * FROM ekstraksi_fitur", cnx)
    x = df[['file', 'energy_0', 'homogenity_0', 'entropy_0', 'contras_0', 'energy_45', 'homogenity_45', 'entropy_45', 'contras_45',
            'energy_90', 'homogenity_90', 'entropy_90', 'contras_90', 'energy_135', 'homogenity_135', 'entropy_135', 'contras_135', 'label']].values
    Z_train,  W_train = train_test_split(x, test_size=0.2, random_state=42)
    uji = W_train
    return uji


@app.route("/import_data_uji")
def import_data_uji():
    uji = datauji()
    print(uji[0])
    for citra in uji:
        files = r"D:/Kampus/SMT8/belajar/sistem-identifikasi-kayu/dataset/" + \
            citra[0]
        gambar = cv2.imread(os.path.abspath(files))
        hasil = r"D:/Kampus/SMT8/belajar/sistem-identifikasi-kayu/uploads/data_uji/" + \
            citra[0]
        cv2.imwrite(hasil, gambar)
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT count(*) FROM uji")
    hasil = np.array(cursor.fetchall())
    jum_data_uji = int(hasil[0][0])
    if jum_data_uji == 0:
        for d_uji in uji:
            cur = mysql.connection.cursor()
            sql = "INSERT INTO uji (file,energy_0,homogenity_0, entropy_0, contras_0,energy_45,homogenity_45, entropy_45, contras_45,energy_90,homogenity_90, entropy_90, contras_90,energy_135,homogenity_135, entropy_135, contras_135, label) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            val = (d_uji[0], d_uji[1], d_uji[2], d_uji[3], d_uji[4],
                   d_uji[5], d_uji[6], d_uji[7], d_uji[8], d_uji[9], d_uji[10], d_uji[11], d_uji[12], d_uji[13], d_uji[14], d_uji[15], d_uji[16], d_uji[17])
            cur.execute(sql, val)
            mysql.connection.commit()
    return redirect(url_for('data_uji'))


@app.route("/data_uji")
def data_uji():
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM uji")
    uji = cursor.fetchall()
    cursor.close
    return render_template('data_uji.html', data=uji)


def spliting_data():
    df = pd.read_sql_query("SELECT * FROM uji", cnx)
    df1 = pd.read_sql_query("SELECT * FROM latih", cnx)
    X = df[['energy_0', 'homogenity_0', 'entropy_0', 'contras_0', 'energy_45', 'homogenity_45', 'entropy_45', 'contras_45',
           'energy_90', 'homogenity_90', 'entropy_90', 'contras_90', 'energy_135', 'homogenity_135', 'entropy_135', 'contras_135']].values
    Y = df['label'].values
    W = df1[['energy_0', 'homogenity_0', 'entropy_0', 'contras_0', 'energy_45', 'homogenity_45', 'entropy_45', 'contras_45',
            'energy_90', 'homogenity_90', 'entropy_90', 'contras_90', 'energy_135', 'homogenity_135', 'entropy_135', 'contras_135']].values
    Z = df1['label'].values
    X_test = X
    y_test = Y
    X_train = W
    y_train = Z
    return X_train, X_test, y_train, y_test


@app.route("/pengujian", methods=["POST", "GET"])
def pengujian():
    df = pd.read_sql_query("SELECT * FROM uji", cnx)
    q_hasil = np.array(df)
    if request.method == 'POST':
        nilai_k = request.form['nilai_k']
        if nilai_k == '':
            return render_template("pengujian.html", menu='klasifikasi', submenu='pengujian', data=q_hasil)

        X_train, X_test, y_train, y_test = spliting_data()
        model = KNN(k=nilai_k)
        model.train(X_train, y_train)
        hasil = model.predict(X_test)
        akurasi = np.sum(hasil == y_test)/len(X_test)
        cm = confusion_matrix(y_test, hasil)
        print(cm)
        benar = int(cm[0][0]) + int(cm[1][1])
        salah = int(cm[0][1]) + int(cm[1][0])

        cls_report = classification_report(y_test, hasil, output_dict=True)
        c_rpt = np.array([
            cls_report["1"]['precision'], cls_report["1"]['recall'], cls_report["1"]['f1-score'], cls_report["1"]['support'],
            cls_report["2"]['precision'], cls_report["2"]['recall'], cls_report["2"]['f1-score'], cls_report["2"]['support'],
            cls_report["macro avg"]['precision'], cls_report["macro avg"]['recall'], cls_report[
                "macro avg"]['f1-score'], cls_report["macro avg"]['support'],
            cls_report["weighted avg"]['precision'], cls_report["weighted avg"][
                'recall'], cls_report["weighted avg"]['f1-score'], cls_report["weighted avg"]['support']
        ])
        akurasi = np.sum(hasil == y_test)/len(X_test)
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM uji")
        id_prt = cursor.fetchall()
        idku = id_prt[0][0]
        for prediksi in hasil:
            cur = mysql.connection.cursor()
            sql = "UPDATE uji SET prediksi = %s WHERE id = %s"
            val = (prediksi, idku)
            cur.execute(sql, val)
            mysql.connection.commit()
            idku += 1
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM uji")
        hasilnya = cursor.fetchall()
        cursor.close
        return render_template("pengujian.html", data=hasilnya,
                               akurasi=round(akurasi, 4), benar=benar, salah=salah,
                               precision1=round(c_rpt[0], 4), recall1=round(c_rpt[1], 4), f1_score1=round(c_rpt[2], 4), support1=round(c_rpt[3], 4),
                               precision2=round(c_rpt[4], 4), recall2=round(c_rpt[5], 4), f1_score2=round(c_rpt[6], 4), support2=round(c_rpt[7], 4),
                               precision3=round(c_rpt[8], 4), recall3=round(c_rpt[9], 4), f1_score3=round(c_rpt[10], 4), support3=round(c_rpt[11], 4),
                               precision4=round(c_rpt[12], 4), recall4=round(c_rpt[13], 4), f1_score4=round(c_rpt[14], 4), support4=round(c_rpt[15], 4))
    return render_template("pengujian.html", data=q_hasil)


@app.route("/prediksi", methods=["POST", "GET"])
def prediksi():
    df = pd.read_sql_query("SELECT * FROM latih", cnx)
    df1 = pd.read_sql_query("SELECT * FROM prediksi", cnx)
    q_hasil = np.array(df)
    prediksi = np.array(df1)
    if request.method == 'POST':
        file = request.files['file']
        nilai_k = request.form['nilai_k']
        X_train, X_test, y_train, y_test = spliting_data()
        model = KNN(k=nilai_k)
        model.train(X_train, y_train)
        if 'file' not in request.files:
            return redirect(url_for('prediksi'))

        if file.filename == '':
            return redirect(url_for('prediksi'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            direc = r"D:/Kampus/SMT8/belajar/sistem-identifikasi-kayu/uploads/data_prediksi/"
            file.save(os.path.join(direc, filename))
            image = cv2.imread(direc+filename)
            if os.path.isfile(direc+filename):
                img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                resize_image = cv2.resize(img, (100, 100))

                derajat_0 = derajat0(resize_image)
                derajat_45 = derajat45(resize_image)
                derajat_90 = derajat90(resize_image)
                derajat_135 = derajat135(resize_image)

                energi_0 = energi(derajat_0)
                homogenity_0 = homogenitas(derajat_0)
                entropy_0 = entropy(derajat_0)
                contras_0 = contras(derajat_0)
                energi_45 = energi(derajat_45)
                homogenity_45 = homogenitas(derajat_45)
                entropy_45 = entropy(derajat_45)
                contras_45 = contras(derajat_45)
                energi_90 = energi(derajat_90)
                homogenity_90 = homogenitas(derajat_90)
                entropy_90 = entropy(derajat_90)
                contras_90 = contras(derajat_90)
                energi_135 = energi(derajat_135)
                homogenity_135 = homogenitas(derajat_135)
                entropy_135 = entropy(derajat_135)
                contras_135 = contras(derajat_135)

                fitur = [
                    round(energi_0, 5),
                    round(homogenity_0, 5),
                    round(entropy_0, 5),
                    round(contras_0, 5),
                    round(energi_45, 5),
                    round(homogenity_45, 5),
                    round(entropy_45, 5),
                    round(contras_45, 5),
                    round(energi_90, 5),
                    round(homogenity_90, 5),
                    round(entropy_90, 5),
                    round(contras_90, 5),
                    round(energi_135, 5),
                    round(homogenity_135, 5),
                    round(entropy_135, 5),
                    round(contras_135, 5)
                ]

                citra = [[
                    fitur[0],
                    fitur[1],
                    fitur[2],
                    fitur[3],
                    fitur[4],
                    fitur[5],
                    fitur[6],
                    fitur[7],
                    fitur[8],
                    fitur[9],
                    fitur[10],
                    fitur[11],
                    fitur[12],
                    fitur[13],
                    fitur[14],
                    fitur[15],
                ]]
                y_pred = model.predict(citra)
                y_pred = int(y_pred[0])

                cur = mysql.connection.cursor()
                sql = "INSERT INTO prediksi (file,energy_0,homogenity_0, entropy_0, contras_0,energy_45,homogenity_45, entropy_45, contras_45,energy_90,homogenity_90, entropy_90, contras_90,energy_135,homogenity_135, entropy_135, contras_135, label) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                val = (filename, fitur[0], fitur[1], fitur[2], fitur[3], fitur[4], fitur[5], fitur[6], fitur[7],
                       fitur[8], fitur[9], fitur[10], fitur[11], fitur[12], fitur[13], fitur[14], fitur[15], y_pred)
                cur.execute(sql, val)
                mysql.connection.commit()

                cursor = mysql.connection.cursor()
                cursor.execute("SELECT * FROM latih")
                id_prt = cursor.fetchall()
                idku = id_prt[0][0]
                jarak = np.array(knn_jarak).flatten()
                for jr in jarak:
                    cur = mysql.connection.cursor()
                    sql = "UPDATE latih SET jarak = %s WHERE id = %s"
                    val = (round(jr, 4), idku)
                    cur.execute(sql, val)
                    mysql.connection.commit()
                    idku += 1
                cursor = mysql.connection.cursor()
                cursor.execute("SELECT * FROM latih")
                hasilnya = cursor.fetchall()
                cursor.close
                cursor = mysql.connection.cursor()
                cursor.execute("SELECT * FROM prediksi")
                prediksi = cursor.fetchall()
                cursor.close
        return render_template("prediksi.html", y_pred=y_pred, data=hasilnya, prediksi=prediksi)
    return render_template("prediksi.html", data=q_hasil, prediksi=prediksi)


if __name__ == "__main__":
    app.run(debug=True)
