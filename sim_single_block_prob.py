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
                sum2 += ((L[n] * Fx[n, l] * By[n, m]) / sum([theta[i, m] * Fx[n, i] for i in range(Mx)]))
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


def sim_single_block(fig, N=1, Mx=3, My=3, Ex=1, Ey=1, algorithms=[]):
    if len(algorithms) == 0: return
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)

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
    theta_evolution = np.zeros((lim, Mx * My))
    for algo in algorithms:
        plt_y = np.zeros((N, lim))
        theta = (1. / My) * np.ones((Mx, My))
        for i in range(lim):
            if algo == "ML": theta = maximum_likelihood(Fx=Fx, By=By, theta=theta, N=N, Mx=Mx, My=My)
            if algo == "KL": theta = minimum_divergence(Fx=Fx, By=By, theta=theta, N=N, Mx=Mx, My=My)
            if algo == "LD": theta = localized_decision(Fx=Fx, By=By, theta=theta, N=N, Mx=Mx, My=My)
            if algo == "VAR": theta = variational(Fx=Fx, By=By, theta=theta, N=N, Mx=Mx, My=My)
            theta = np.divide(theta, np.linalg.norm(theta, ord=1, axis=1).reshape(Mx,1))
            theta_evolution[i] = theta.ravel()
        for i in range(Mx*My):
            if algo == "ML": ax1.plot(theta_evolution[:, i].T)
            if algo == "KL": ax2.plot(theta_evolution[:, i].T)
            if algo == "LD": ax3.plot(theta_evolution[:, i].T)
            if algo == "VAR": ax4.plot(theta_evolution[:, i].T)
    ax1.set_title("ML")
    ax2.set_title("KL")
    ax3.set_title("LD")
    ax4.set_title("VAR")
if __name__ == '__main__':
    fig = plt.figure()
    sim_single_block(fig, N=60, Mx=5, My=3, Ex=1, Ey=1, algorithms= ["ML", "KL", "LD", "VAR"])
    plt.suptitle("Evolution of $\\theta$ N=60, Mx=5, My=3, Ex=1, Ey=1")
    plt.show()
    # plt.savefig("evolution_of_conditional_prob1.png")

    sim_single_block(fig, N=300, Mx=7, My=9, Ex=10, Ey=10, algorithms= ["ML", "KL", "LD", "VAR"])
    plt.suptitle("Evolution of $\\theta$ N=300, Mx=7, My=9, Ex=10, Ey=10")
    plt.show()
    # plt.savefig("evolution_of_conditional_prob2.png")

