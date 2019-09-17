import time
import numpy as np

from algorithm_ncs.benchmark import benchmark_func
from algorithm_ncs.problem import load_problem
from algorithm_ncs import *

"""
I was crashed!!!
"""
class ProblemResult:
    def __init__(self, x, fit):
        self.x = x
        self.fit = fit
        self.corr = None

        self.newX = None
        self.newFit = None
        self.newCorr = None


def ncs(problem, para: SuperParameter, filter=True):
    """
    choose the problems to be tested. Please note that for test functions F7
    and F25, the global optima are out of the initialization range. For these
    two test functions, we do not need to check whether the variable violates
    the boundaries during the evolution after the initialization.

    :param problem_index:
    :param para: super-parameter for NCS algorithm
    :param filter: if it is problem 7 or 25
    :return: the final solution
    """
    # pre load the data from the problem
    D = 30  # the dimension of the problem
    parameters = load_problem(problem, D)
    mu = 4 + np.math.floor(3 * np.log(D))  # the number of solutions in each search process

    # Set the randomness seed
    np.random.seed(int(time.time()))

    # Configuration of NCS
    N = para.N  # the number of search processes, also is the number of solution in a population
    Tmax = para.Tmax
    sigma = [para.sigma*abs(parameters.lu[0][1]-parameters.lu[0][0])] * N

    lu = parameters.lu
    vl = np.tile(lu[:, 0].reshape((-1, 1)), (1, mu))
    vu = np.tile(lu[:, 1].reshape((-1, 1)), (1, mu))

    # Definition of the structure of search processes
    sp = [
        ProblemResult(
            x=np.random.random((D, mu)),
            fit=np.ones(mu)) for _ in range(N)
    ]

    # Re-initialize the best solution recorder in this run
    min_f = 1e300
    bestS = None

    c = [0] * N

    for i in range(0, N):
        sp[i].fit = benchmark_func(sp[i].x,
                                   problem,
                                   parameters.o,
                                   parameters.A,
                                   parameters.M,
                                   parameters.a,
                                   parameters.alpha,
                                   parameters.b)
        if sp[i].fit[0] < min_f:
            min_f = sp[i].fit[0]
            bestS = sp[i].x

    t = 0
    # The main loop body
    while t < Tmax:

        _lambda = np.random.normal(1, 0.1 - 0.1 * t / Tmax)

        for i in range(0, N):
            sp[i].newX = sp[i].x + np.random.normal(0, sigma[i], size=sp[i].x.shape)
            # Generate mu solutions for each search process
            # sp[i].x = np.tile(sp[i].mean, (1, mu)) + np.random.normal(0, 1, size=(D, mu)) * (np.tile(sp[i].cov, (1, mu)))
            # Boundary checking and repairing
            # problem != 7 and problem != 25:
            if filter:
                pos = sp[i].newX < vl
                sp[i].newX[pos] = 2 * vl[pos] - sp[i].newX[pos]
                pos = sp[i].newX > vu
                sp[i].newX[pos] = 2 * vu[pos] - sp[i].newX[pos]
                pos = sp[i].newX < vl
                sp[i].newX[pos] = vl[pos]

            sp[i].newFit = benchmark_func(sp[i].newX,
                                          problem,
                                          parameters.o,
                                          parameters.A,
                                          parameters.M,
                                          parameters.a,
                                          parameters.alpha,
                                          parameters.b)

            min_corr = 1e300
            min_corr_new = 1e300
            for j in range(0, N):
                if i == j:
                    continue

                SIGMA_i = np.linalg.inv(np.eye(D) * sigma[i] ** 2)
                SIGMA_j = np.linalg.inv(np.eye(D) * sigma[j] ** 2)
                SIGMA_inv = np.linalg.inv((SIGMA_i + SIGMA_j) / 2)
                Db_ij = np.dot(
                    np.mean((sp[i].x - sp[j].x).T, 0),
                    np.dot(SIGMA_inv,  np.mean((sp[i].x - sp[j].x), 1))
                ) / 8 + D * np.math.log1p((sigma[i] ** 2 + sigma[j] ** 2) / (2 * sigma[i] * sigma[j])) / 2
                Db_ij_new = np.dot(
                    np.mean((sp[i].newX - sp[j].x).T, 0),
                    np.dot(SIGMA_inv, np.mean(sp[i].newX - sp[j].x, 1))
                ) / 8 + D * np.math.log1p((sigma[i] ** 2 + sigma[j] ** 2) / (2 * sigma[i] * sigma[j])) / 2
                min_corr = min(min_corr, Db_ij)
                min_corr_new = min(min_corr_new, Db_ij_new)

            sp[i].newCorr = min_corr_new
            sp[i].corr = min_corr

        for i in range(0, N):

            min_temp = min(sp[i].newFit)
            if min_temp < min_f:
                min_f = min_temp
                bestS = sp[i].newX
                print("best 1e%.2f (%.2f, %.2f) time: %d" % (np.log(min_f), bestS.max(), bestS.min(), t))

            # normalize
            newFit = np.mean(sp[i].newFit) / (np.mean(sp[i].newFit) + np.mean(sp[i].fit))
            newCorr = np.mean(sp[i].newCorr) / (np.mean(sp[i].newCorr) + np.mean(sp[i].corr))

            if newFit / newCorr < _lambda:
                sp[i].x = sp[i].newX
                sp[i].corr = sp[i].newCorr
                c[i] = c[i] + mu

        t = t + mu
        if t % para.epoch == 0:
            for i in range(N):
                if c[i] / para.epoch > 0.2:
                    sigma[i] = sigma[i] / para.r
                elif c[i] / para.epoch < 0.2:
                    sigma[i] = sigma[i] * para.r
                c[i] = 0

    return min_f



if __name__ == '__main__':
    problem_set = [6, 12]
    for p in problem_set:
        print("\n************ the problem %d started! ************\n" % p)

        total_time = 25
        outcome = np.ones(total_time)
        super_para = SuperParameter(Tmax=300000, N=10, sigma=0.1, r=0.99, epoch=10)
        for i in range(total_time):
            start = time.time()
            if p == 7 or p == 25:
                out = ncs(p, super_para, False)
            else:
                out = ncs(p, super_para)
            outcome[i] = out
            print('the {} th problem cost{}: {}, res: {}'.format(p, i, time.time() - start, out))

        print('the {} th problem result is:'.format(p))
        print('the mean result is: {} and the std is {}'.format((np.mean(outcome)), np.std(outcome)))
