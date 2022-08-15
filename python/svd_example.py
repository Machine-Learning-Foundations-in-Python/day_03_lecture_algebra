import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    A = np.random.randn(3, 4)

    sigma1, V = np.linalg.eig(A.T@A)
    sigma2, U = np.linalg.eig(A@A.T)

    U2, sigma3, V2 = np.linalg.svd(A)

    sig_mat = np.zeros(A.shape)
    # sig_mat[:4, :4] = np.diag(1./sigma3)
    sig_mat[:A.shape[0], :A.shape[0]] = np.diag(sigma3)
    A_rec = U2@sig_mat@V2 

    A_pinv = np.linalg.pinv(A)
    sig_mat_inv = np.zeros(A.T.shape)
    sig_mat_inv[:A.shape[0], :A.shape[0]] = np.diag(sigma3**(-1))
    A_inv = V2.T@sig_mat_inv@U2.T


    print('done')