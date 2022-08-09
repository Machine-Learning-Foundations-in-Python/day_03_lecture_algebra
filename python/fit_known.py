from cProfile import label
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    b = np.array([0., 0.5, 1., 0.75, 1.5, 2., 0.])
    # b = np.expand_dims(b, -1)
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
    b_noise = b_true + alpha*np.random.randn(point_no)

    # plt.plot(x_axis2, b_true)
    # plt.plot(x_axis2, b_noise)
    # plt.show()

    A3 = np.zeros((point_no, point_no))

    for i in range(point_no):
        A3[:, i] = x_axis2**i

    # regularize to find the noise free solution.
    U, sigma, V = np.linalg.svd(A3)
    plt.plot(x_axis2, b_noise)

    for scale in [1e-2, 1e-5, 0.]:
        # x = np.zeros(point_no)
        # for i in range(point_no):
        #     f = sigma[i]**2 / (sigma[i]**2 + scale) 
        #     x = x + f*((U[:,i].T @ b_noise)/sigma[i]) * V.T[:,i]  
        fVec = sigma**2 / (sigma**2 + scale)
        F = np.diag(fVec)
        # x2 = np.sum(V.T @ np.diag(U.T @ b_noise / sigma)@F, 1) 
        x2 = V.T @ F @ np.diag(sigma**(-1)) @ U.T @ b_noise

        plt.plot(x_axis2, A3@x2, label=str(scale))
    plt.legend()
    plt.ylim(-1., 3.)
    plt.grid()
    plt.show()
    print('stop')
