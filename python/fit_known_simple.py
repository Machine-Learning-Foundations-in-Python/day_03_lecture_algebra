import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    np.random.seed(0)
    b = np.array([0., 0.5, 1., 0.75, 1.5, 2., 0.])
    x_axis = np.linspace(0, 1, num=7)

    A1 = np.zeros((7, len(b)))
    for i in range(0, len(b)):    
        A1[:, i] = x_axis**i
    
    est_x = np.linalg.pinv(A1)@b

    point_no = 300
    x_axis2 = np.linspace(0, 1, num=point_no)
    A2 = np.zeros((point_no, len(b)))
    for i in range(0, len(b)):    
        A2[:, i] = x_axis2**i

    alpha = 0.2
    b_true = A2@np.linalg.pinv(A1)@b

    point_no = 300
    x_axis2 = np.linspace(0, 1, num=point_no)
    A3 = np.zeros((point_no, 2))
    for i in range(0, 2):    
        A3[:, i] = x_axis2**i

    xb = np.linalg.pinv(A3)@b_true
    est = x_axis2*xb[1] + xb[0]

    plt.plot(x_axis2, b_true)
    plt.plot(x_axis2, est)
    plt.show()
