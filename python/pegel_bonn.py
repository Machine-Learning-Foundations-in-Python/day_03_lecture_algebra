# using data from https://pegel.bonn.de/php/rheinpegel.php

from sqlite3 import Timestamp
import matplotlib.pyplot as plt
import numpy as np
import pandas
from datetime import datetime, date

if __name__ == "__main__":
    rhein = pandas.read_csv('./python/pegel.tab', sep=" 	")
    levels = np.array([int(pegel.split(" ")[0]) for pegel in rhein['Pegel']])

    timestamps = [ts[:-4] for ts in rhein['Zeit']]
    datetime_list = []
    keep_indices = []
    for idx, ts in enumerate(timestamps):
        ts_date, ts_time = ts.split(",")
        day, month, year = ts_date.split(".")
        hour, minute = ts_time.split(":")
        datetime_list.append(datetime(
            int(year), int(month), int(day)))
        if hour == ' 04' or hour == ' 05':
            keep_indices.append(idx)

    
    levels = levels[keep_indices]
    datetime_list = np.array(datetime_list)[keep_indices]

    datetime_stamps = [dt.timestamp() for dt in datetime_list]

    levels = levels[::-1]
    datetime_stamps = datetime_stamps[::-1]
    datetime_list = datetime_list[::-1]

    plt.plot(datetime_stamps, levels)
    plt.show()

    point_no = len(levels)
    # x_axis = np.linspace(0, 1, num=point_no)
    x_axis = np.array(datetime_stamps)
    A3 = np.zeros((point_no, 2))
    for i in range(0, 2):    
        A3[:, i] = x_axis**i

    xb = np.linalg.pinv(A3)@levels
    est = x_axis*xb[1] + xb[0]

    plt.plot(datetime_list, levels)
    plt.plot(datetime_list, est)
    plt.xlabel("Jahr")
    plt.ylabel("Pegel [cm]")
    plt.title('Rheinpegel bei Bonn')
    plt.show()
 
    zero = -xb[0]/xb[1]
    print("zero:", datetime.fromtimestamp(zero))

    x_axis = np.linspace(0, 1, len(datetime_list))
    degree = 20
    A4 = np.zeros((point_no, degree))
    for i in range(0, degree):    
        A4[:, i] = x_axis**i

    xb = np.linalg.pinv(A4)@levels
    est4 = A4@xb

    plt.plot(datetime_list, levels)
    plt.plot(datetime_list, est4)
    plt.title("Rheinpegel bei Bonn")
    plt.show()

    plt.plot(datetime_list, levels)

    U, sigma, V = np.linalg.svd(A4)

    for scale in [1e-4, 1e-9, 0.]:
        # x = np.zeros(point_no)
        # for i in range(point_no):
        #     f = sigma[i]**2 / (sigma[i]**2 + scale) 
        #     x = x + f*((U[:,i].T @ b_noise)/sigma[i]) * V.T[:,i]  
        fVec = sigma**2 / (sigma**2 + scale)
        F = np.diag(fVec)
        S = np.zeros((degree, len(levels)))
        S[:degree, :degree] = np.diag(sigma**(-1))
        # x2 = np.sum(V.T @ np.diag(U.T @ b_noise / sigma)@F, 1) 
        x_reg = V.T @ F @ S @ U.T @ levels

        plt.plot(datetime_list, A4@x_reg, label=str(scale))
    plt.legend()
    # plt.ylim(-1., 3.)
    plt.title("Rheinpegel bei Bonn")
    plt.ylabel("Pegel cm")
    plt.xlabel("Jahr")
    plt.show()

    print('done')