import numpy as np
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
import pandas as pd
from scipy.fft import fft, ifft
import math
from scipy.stats.stats import pearsonr
import eeglib
from scipy.stats import kurtosis, skew
from scipy import stats
import os
import pickle
import time
import argparse
import subprocess
import sys
import re
import os
from csv import writer, reader
from pythonosc import osc_server
import secrets

secret = secrets.token_urlsafe(32)


app = Flask(__name__)
app.secret_key = secret
model = pickle.load(open('RF_model_t.pkl', 'rb'))

@app.route('/')
def home():
    flash('App Started')
    return render_template('index.html')

@app.route('/stream',methods=['POST'])
def stream():
    flash('You were successfully logged in')
    def get_pids(port):

        command = "sudo lsof -i :%s | awk '{print $2}'" % port
        pids = subprocess.check_output(command, shell=True)
        pids = pids.strip()
        if pids:
            pids = re.sub(' +', ' ', pids)
            for pid in pids.split('\n'):
                try:
                    yield int(pid)
                except:
                    pass

    if __name__ == '__main__':

        port = 8000
        pids = set(get_pids(port))
        command = 'sudo kill -9 {}'.format(' '.join([str(pid) for pid in pids]))
        os.system(command)
        print(command)
    filename = 'realtimeeeg.csv'
    f = open(filename, "w+")
    f.close()
    head = []
    head.append('RAW_TP10')
    head.append('RAW_AF7')
    head.append('RAW_AF8')
    head.append('RAW_TP9')
    with open(filename, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(head)

    def eeg_handler(unused_addr, args, arr, ch1, ch2, ch3, ch4):
        with open(r'realtimeeeg.csv', 'a+', newline='') as write_obj:
            # print(ch1,ch2,ch3,ch4)
            row = []
            row.append(ch1)
            row.append(ch2)
            row.append(ch3)
            row.append(ch4)
            csv_writer = writer(write_obj)
            csv_writer.writerow(row)

    if __name__ == '__main__':
        Port = 8000
        parser = argparse.ArgumentParser()
        parser.add_argument("--ip", default="0.0.0.0")
        parser.add_argument("--port", default=8000)
        args = parser.parse_args()
        from pythonosc import dispatcher
        dispatcher = dispatcher.Dispatcher()
        dispatcher.map("/debug", print)
        dispatcher.map("/muse/eeg", eeg_handler, "EEG")

        server = osc_server.ThreadingOSCUDPServer((args.ip, args.port), dispatcher)
        server.serve_forever()


@app.route('/predict',methods=['POST'])
def predict():
    para = 0
    print("predicting")
    f = open('test.csv', "w+")
    f.close()
    f = open('test.csv', "w+")
    f.close()
    df1 = pd.read_csv("realtimeeeg.csv")
    baseidx = np.zeros(2)
    baseidx[0] = 125
    baseidx[1] = 256
    df = df1[0:256]

    # continuous wavelet transform
    # wavelet parameters
    [m, n] = df.shape
    num_frex = 30
    min_freq = 8
    max_freq = 27
    srate = 256
    frex = np.linspace(min_freq, max_freq, num_frex)
    time1 = np.arange(-1.5, 1.5, 1 / srate)
    half_wave = round((len(time1) - 1) / 2)
    # FFT parameters
    nKern = len(time1)
    nData = m
    nConv = nKern + nData - 1
    # initialize output time1-frequency data

    # baseidx[0] = find_nearest(df['time1'], baseline_window[0])
    # baseidx[1] = find_nearest(df['time1'], baseline_window[1])
    baseidx = baseidx.astype(int)
    tf = [[[0] * m] * len(frex)] * 4
    tf = np.asarray(tf, dtype=float)
    channels = ['RAW_TP10', 'RAW_AF7', 'RAW_AF8', 'RAW_TP9']
    for cyclei in range(0, 4):
        dataX = fft(df[channels[cyclei]].to_numpy(), nConv)
        for i in range(0, len(frex)):
            s = 8 / (2 * math.pi * frex[i])
            cmw = np.multiply(np.exp(np.multiply(2 * complex(0, 1) * math.pi * frex[i], time1)),
                              np.exp(np.divide(-time1 ** 2, (2 * s ** 2))))
            cmwX = fft(cmw, nConv)
            cmwX = np.divide(cmwX, max(cmwX))
            as1 = ifft(np.multiply(cmwX, dataX), nConv)
            as1 = as1[half_wave:len(as1) - half_wave + 1]
            as1 = np.reshape(as1, m);
            mag = np.absolute(as1) ** 2
            tf[cyclei, i, :] = np.absolute(as1) ** 2;
        # print((np.squeeze(tf[cyclei,:,:])/ np.transpose(np.mean(tf[cyclei,:,baseidx[0]:baseidx[1]],1))))
        var = np.transpose(np.mean(tf[cyclei, :, baseidx[0]:baseidx[1]], 1))
    rows = len(df1.index)

    while (1):
        if (df.empty):
            print("No data present because of no connection")
            end = time.time()

            break
        if (rows > 256):
            df = df1[para * 26:(para + 1) * 26]
            # continuous wavelet transform
            # wavelet parameters
            [m, n] = df.shape
            num_frex = 30
            min_freq = 8
            max_freq = 27
            srate = 256
            frex = np.linspace(min_freq, max_freq, num_frex)
            time1 = np.arange(-1.5, 1.5, 1 / srate)
            half_wave = round((len(time1) - 1) / 2)
            # FFT parameters
            nKern = len(time1)
            nData = m
            nConv = nKern + nData - 1
            # initialize output time1-frequency data

            # baseidx[0] = find_nearest(df['time1'], baseline_window[0])
            # baseidx[1] = find_nearest(df['time1'], baseline_window[1])
            baseidx = baseidx.astype(int)
            tf = [[[0] * m] * len(frex)] * 4
            tf = np.asarray(tf, dtype=float)
            channels = ['RAW_TP10', 'RAW_AF7', 'RAW_AF8', 'RAW_TP9']
            for cyclei in range(0, 4):
                dataX = fft(df[channels[cyclei]].to_numpy(), nConv)
                for i in range(0, len(frex)):
                    s = 8 / (2 * math.pi * frex[i])
                    cmw = np.multiply(np.exp(np.multiply(2 * complex(0, 1) * math.pi * frex[i], time1)),
                                      np.exp(np.divide(-time1 ** 2, (2 * s ** 2))))
                    cmwX = fft(cmw, nConv)
                    cmwX = np.divide(cmwX, max(cmwX))
                    as1 = ifft(np.multiply(cmwX, dataX), nConv)
                    as1 = as1[half_wave:len(as1) - half_wave + 1]
                    as1 = np.reshape(as1, m);
                    mag = np.absolute(as1) ** 2
                    tf[cyclei, i, :] = np.absolute(as1) ** 2;
                # print((np.squeeze(tf[cyclei,:,:])/ np.transpose(np.mean(tf[cyclei,:,baseidx[0]:baseidx[1]],1))))
                # var = np.transpose(np.mean(tf[cyclei, :, baseidx[0]:baseidx[1]], 1))
                # print(var)

                tf[cyclei, :, :] = 10 * np.log10(np.divide((np.squeeze(tf[cyclei, :, :])).T, var).T)

            # tf=tf[:,1:len(frex),:]
            #        pts = np.where(np.logical_and(df['time1'] >= 0.0, df['time1'] <= 4.0))
            #       pts = np.asarray(pts)
            #      [m1, n1] = pts.shape
            #      tf = tf[:, :, pts]
            #     tf = np.reshape(tf, (4, len(frex), n1))

            falpha = np.where(np.logical_and(frex >= 8, frex <= 13))
            falpha = np.asarray(falpha)
            tfalpha = [[0] * m] * 4
            tfalpha = np.array(tfalpha, dtype=float)
            for i in range(4):
                tfalpha[i] = np.mean(tf[i, falpha, :], axis=1)

            fbeta = np.where(np.logical_and(frex >= 13, frex <= 27))
            fbeta = np.asarray(fbeta)
            tfbeta = [[0] * m] * 4
            tfbeta = np.array(tfbeta, dtype=float)
            for i in range(4):
                tfbeta[i] = np.mean(tf[i, fbeta, :], axis=1)



            features = np.zeros(shape=78)
            mobilityalpha = np.zeros(shape=4)
            mobilitybeta = np.zeros(shape=4)
            j = 0

            for i in range(4):
                features[j] = np.mean(tfalpha[i, :])
                j = j + 1

            for i in range(4):
                features[j] = np.mean(tfbeta[i, :])
                j = j + 1

            for i in range(4):
                features[j] = np.var(tfalpha[i, :])
                j = j + 1

            for i in range(4):
                features[j] = np.var(tfbeta[i, :])
                j = j + 1

            features[j] = features[0] + features[2] - (features[1] + features[3])
            j = j + 1;
            features[j] = features[4] + features[6] - (features[5] + features[7])
            j = j + 1;

            for i in range(4):
                for l in range(4):
                    if i > l:
                        features[j] = pearsonr(np.transpose(tfbeta[i, :]), np.transpose(tfbeta[l, :]))[0]
                        j = j + 1
                        features[j] = pearsonr(np.transpose(tfalpha[i, :]), np.transpose(tfalpha[l, :]))[0]
                        j = j + 1
                        features[j] = pearsonr(np.transpose(tfbeta[i, :]), np.transpose(tfalpha[l, :]))[0]
                        j = j + 1
                        features[j] = pearsonr(np.transpose(tfbeta[l, :]), np.transpose(tfalpha[i, :]))[0]
                        j = j + 1
                    if i == l:
                        features[j] = pearsonr(np.transpose(tfbeta[i, :]), np.transpose(tfalpha[l, :]))[0]
                        j = j + 1

            for i in range(4):
                features[j] = eeglib.features.hjorthMobility(tfalpha[i, :])
                j = j + 1

            for i in range(4):
                features[j] = eeglib.features.hjorthMobility(tfbeta[i, :])
                j = j + 1

            for i in range(4):
                features[j] = eeglib.features.hjorthComplexity(tfalpha[i, :])
                j = j + 1

            for i in range(4):
                features[j] = eeglib.features.hjorthComplexity(tfbeta[i, :])
                j = j + 1

            for i in range(4):
                features[j] = skew(tfbeta[i, :])
                j = j + 1

            for i in range(4):
                features[j] = skew(tfalpha[i, :])
                j = j + 1

            for i in range(4):
                features[j] = kurtosis(tfalpha[i, :])
                j = j + 1

            for i in range(4):
                features[j] = kurtosis(tfbeta[i, :])
                j = j + 1
            para += 1

            with open(r'test.csv', 'a+', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow(features)
        if (para >= 3):
            break

    if (os.stat("test.csv").st_size == 0):
        print("Connection not made with the device")
        color="Connection not made with the device"

    else:
        df = pd.read_csv('test.csv', header=None)
        if (len(df.index) == 1):
            df.to_csv(r'test2.csv', index=False, header=None)
        elif (len(df.index) > 1):
            df1 = stats.zscore(df)
            df1 = pd.DataFrame(data=df1)
            df1.to_csv(r'test2.csv', index=False, header=None)
        import joblib
        #model = joblib.load("/Users/mahima/Downloads/RF_model_t.pkl")
        colors = []
        line1 = []
        file1 = open('test2.csv', 'r')
        Lines = file1.readlines()
        for line in Lines:
            line = line.strip('\n')
            line1 = []
            for y in line.split(','):
                line1.append(float(y))
            line = np.array(line1)
            line = line.reshape(1, -1)

            y = model.predict(line)

            colors.append(y)

        color = max(colors, key=colors.count)

        if color == 1:
            color = "Red"
        elif color == 2:
            color = "Green"
        elif color == 3:
            color = "Blue"
        # print(color)
    def get_pids(port):

        command = "sudo lsof -i :%s | awk '{print $2}'" % port
        pids = subprocess.check_output(command, shell=True)
        pids = pids.strip()
        if pids:
            pids = re.sub(' +', ' ', pids)
            for pid in pids.split('\n'):
                try:
                    yield int(pid)
                except:
                    pass

    if __name__ == '__main__':

        port = 8000
        pids = set(get_pids(port))
        command = 'sudo kill -9 {}'.format(' '.join([str(pid) for pid in pids]))
        print(command)
        os.system(command)
    filename = 'realtimeeeg.csv'
    f = open(filename, "w+")
    f.close()
    head = []
    head.append('RAW_TP10')
    head.append('RAW_AF7')
    head.append('RAW_AF8')
    head.append('RAW_TP9')
    with open(filename, 'w', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(head)
    return render_template('index.html', prediction_text='The result is {}'.format(color))


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=8080)

