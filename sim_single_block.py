#!usr/bin/python
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


def maximum_likelihood(Fx, By, theta, N=1, Mx=3, My=3):
    """
    Implementing alogo. 1 from Table 1
    :param N: 
    :param Mx: 
    :param My: 
    :param Ex: 
    :param Ey: 
    :return: the ML quantity for each distribution
    """

    L = np.ones(N)
    for l in range(Mx):
        sum1 = 0
        for n in range(N):
            sum1 += L[n] * Fx[n, l]
        for m in range(My):
            sum2 = 0
            for n in range(N):
                sum2 += ((L[n] * Fx[n, l] * By[n, m]) / Fx[n].T.dot(theta).dot(By[n]))
            theta[l, m] = (theta[l, m] / sum1) * sum2
    return theta


def minimum_divergence(Fx, By, theta, N=1, Mx=3, My=3):
    """
    Implementing alogo. 1 from Table 1
    :param N: 
    :param Mx: 
    :param My: 
    :param Ex: 
    :param Ey: 
    :return: the ML quantity for each distribution
    """
    L = np.ones(N)
    for l in range(Mx):
        sum1 = 0
        for n in range(N):
            sum1 += L[n] * Fx[n, l]
        for m in range(My):
            sum2 = 0
            for n in range(N):
                sum2 += ((L[n] * Fx[n, l] * By[n    , m]) / sum([theta[i, m] * Fx[n, i] for i in range(Mx)]))
            theta[l, m] = (theta[l, m] / sum1) * sum2
    return theta


def localized_decision(Fx, By, theta, N=1, Mx=3, My=3):
    """
    Implementing alogo. 1 from Table 1
    :param N: 
    :param Mx: 
    :param My: 
    :param Ex: 
    :param Ey: 
    :return: the ML quantity for each distribution
    """
    def IMax(p):
        x = np.zeros((len(p), 1))
        x[np.argmax(p)] = 1.
        return x

    delta = 3.0
    L = np.ones(N)
    ex = np.zeros((N, Mx, 1))
    ey = np.zeros((N, My, 1))
    theta = np.zeros((Mx, My))
    for n in range(N):
        ex[n] = IMax(Fx[n]) + delta * np.ones((Mx, 1))
        ey[n] = IMax(By[n]) + delta * np.ones((My, 1))
        theta += (L[n] * ex[n].dot(ey[n].T))
    return theta


def variational(Fx, By, theta, N=1, Mx=3, My=3):
    """
    Implementing alogo. 1 from Table 1
    :param N: 
    :param Mx: 
    :param My: 
    :param Ex: 
    :param Ey: 
    :return: the ML quantity for each distribution
    """
    delta = 100.0
    L = np.ones(N)
    for l in range(Mx):
        for m in range(My):
            theta[l,m] = delta + np.sum([(L[n] * Fx[n, l] * By[n, m]) for n in range(N)])
    return theta


def sim_single_block(N=1, Mx=3, My=3, Ex=1, Ey=1, algorithms=[]):
    if len(algorithms) == 0: return
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # ax2 = fig.add_subplot(122)

    x = np.random.random(Mx)
    ux = x / np.linalg.norm(x, ord=1)
    y = np.random.random(My)
    uy = y / np.linalg.norm(y, ord=1)

    Fx = np.random.rand(N, Mx)
    By = np.random.rand(N, My)

    for n in range(N):
        Fx[n] = ux ** Ex  # letting the proportional constant be 1
        Fx[n] /= np.linalg.norm(Fx[n], ord=1)
        By[n] = uy ** Ey  # letting the proportional constant be 1
        By[n] /= np.linalg.norm(By[n], ord=1)

    # for i in range(Mx):
    #     ax2.plot(Fx[:, i].T, label="fx_{}".format(i))
    # for i in range(My):
    #     ax2.plot(By[:, i].T, label="by_{}".format(i))

    lim = 100
    for algo in algorithms:
        plt_y = np.zeros((N, lim))
        theta = (1. / My) * np.ones((Mx, My))
        for i in range(lim):
            if algo == "ML": theta = maximum_likelihood(Fx=Fx, By=By, theta=theta, N=N, Mx=Mx, My=My)
            if algo == "KL": theta = minimum_divergence(Fx=Fx, By=By, theta=theta, N=N, Mx=Mx, My=My)
            if algo == "LD": theta = localized_decision(Fx=Fx, By=By, theta=theta, N=N, Mx=Mx, My=My)
            if algo == "VAR": theta = variational(Fx=Fx, By=By, theta=theta, N=N, Mx=Mx, My=My)
            theta = np.divide(theta, np.linalg.norm(theta, ord=1, axis=1).reshape(Mx,1))
            for n in range(N):
                plt_y[n, i] = np.log(Fx[n, :].dot(theta).dot(By[n, :].T))
        res = np.sum(plt_y, axis=0)
        # ax1.plot(plt_y[N-1], label = algo)
        ax1.plot(res, label=algo)
    ax1.set_xlabel("epoque")
    ax1.legend()
    ax1.set_title("N={}, Mx={}, My={}, Ex={}, Ey={}".format(N, Mx, My, Ex, Ey))
    plt.legend()
    plt.show()
    # plt.savefig("evolution_of_loglikelyhood_N_{}_Mx_{}_My_{}_Ex_{}_Ey_{}.png".format(N,Mx, My, Ex, Ey))

if __name__ == '__main__':
    sim_single_block(N=60, Mx=4, My=3, Ex=5, Ey=5, algorithms= ["ML", "KL", "LD", "VAR"])
    # sim_single_block(N=60, Mx=7, My=9, Ex=5, Ey=5, algorithms=["ML", "KL", "LD", "VAR"])
    # sim_single_block(N=100, Mx=5, My=10, Ex=1, Ey=1, algorithms=["ML", "KL", "LD", "VAR"])
    # sim_single_block(N=300, Mx=7, My=9, Ex=10, Ey=10, algorithms=["ML", "KL", "LD", "VAR"])
