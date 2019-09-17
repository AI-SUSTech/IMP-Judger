import math
import numpy as np
import time

from algorithm_ncs.benchmark import benchmark_func
from algorithm_ncs.problem import load_problem
from algorithm import *


class ProblemResult:
    def __init__(self, x, fit, mean, cov):
        self.x = x
        self.fit = fit
        self.mean = mean
        self.cov = cov


def ncs(problem_index, super_para: SuperParameter, filter=True):
    """
    :param problem_index:
    :param filter: if it is problem 7 or 25
    :param total_time: the total number of runs
    :return:
    """
    # pre load the data from the problem
    D = 30  # the dimension of the problem
    parameters = load_problem(problem_index, D)

    # Configuration of NCS
    _lambda = math.ceil(np.log(D))  # the number of search processes
    N = super_para.N
    mu = 4 + math.floor(3 * np.log(D))  # the number of solutions in each search process
    phi_init = 0.00005  # the trade-off parameter between f and d
    eta_m_init = 1  # the step size of gradient descent for mean vector
    eta_c_init = (3 + np.log(D)) / (5 * np.sqrt(D))  # the step size of gradient descent for mean vector

    lu = parameters.lu
    vl = np.tile(lu[:, 0].reshape((-1, 1)), (1, mu))
    vu = np.tile(lu[:, 1].reshape((-1, 1)), (1, mu))

    # Configuration of the test protocol
    MAXFES = 10000 * D  # the total FE of each run

    # Record the best results for eachnp.mean( problem
    outcome = 1e300

    # Definition of the structure of search processes
    sp = [
        ProblemResult(
            x=parameters.o,  # np.zeros((D, mu)),
            fit=np.zeros(mu),
            mean=np.zeros((D, 1)),
            cov=np.zeros((D, 1))) for _ in range(N)]

    # Set the randomness seed
    np.random.seed(int(time.time()))

    # Re-initialize the best solution recorder in this run
    min_f = 1e300
    FES = 0

    for i in range(0, N):
        # Model the search process as Gaussian probabilistic distribution
        sp[i].mean = lu[:, 0].reshape((-1, 1)) + np.random.rand(D, 1) * (lu[:, 1] - lu[:, 0]).reshape((-1, 1))
        sp[i].cov = (lu[:, 1] - lu[:, 0]).reshape((-1, 1)) / N

    # The main loop body
    while FES < MAXFES:

        eta_m = eta_m_init * ((math.exp(1) - math.exp(FES / MAXFES)) / (math.exp(1) - 1))
        eta_c = eta_c_init * ((math.exp(1) - math.exp(FES / MAXFES)) / (math.exp(1) - 1))

        for i in range(0, N):
            # Generate mu solutions for each search process
            sp[i].x = np.tile(sp[i].mean, (1, mu)) + np.random.normal(0, 1, size=(D, mu)) * (
                np.tile(sp[i].cov, (1, mu)))
            # Boundary checking and repairing
            # problem != 7 and problem != 25:
            if filter:
                # todo < in matlab is different in python
                pos = sp[i].x < vl
                sp[i].x[pos] = 2 * vl[pos] - sp[i].x[pos]
                pos = sp[i].x > vu
                sp[i].x[pos] = 2 * vu[pos] - sp[i].x[pos]
                pos = sp[i].x < vl
                sp[i].x[pos] = vl[pos]

                # Fitness evalution for mu solutions
            sp[i].fit = benchmark_func(sp[i].x,
                                       problem_index,
                                       parameters.o,
                                       parameters.A,
                                       parameters.M,
                                       parameters.a,
                                       parameters.alpha,
                                       parameters.b)
            FES = FES + mu

            # Update the best solution ever found
            temp_min_f = min(sp[i].fit)
            if temp_min_f < min_f:
                min_f = temp_min_f
                print("best 1e%.2f (%d/%d)" % (np.log(min_f), FES, MAXFES))

            # Rank mu solutions ascendingly in terms of fitness
            order = np.argsort(sp[i].fit)
            rank = np.argsort(order) + 1

            # Set utility value for mu solutions in terms of rank
            tempU = [i if i > 0 else 0 for i in np.log(mu / 2 + 1) - np.log(rank.T)]
            utility = np.divide(tempU, sum(tempU)) - 1 / mu

            # Prepare for calculating gradients (for saving computation current_time)
            invCov_i = np.divide(1, sp[i].cov)
            difXtoMean = sp[i].x - np.tile(sp[i].mean, reps=(1, mu))

            # Calculate the gradients of expectation of fitness values
            deltaMean_f = invCov_i * np.mean(
                difXtoMean * np.tile(utility, reps=(D, 1)), 1).reshape((D, 1))  # w.r.t. mean vector
            deltaCov_f = np.power(invCov_i, 2) * np.mean(
                np.power(difXtoMean, 2) * np.tile(utility, reps=(D, 1)), 1).reshape(
                (-1, 1)) / 2  # w.r.t. covariance matrix

            # Calculate the gradients of distribution distances
            deltaMean_d = np.zeros((D, 1))  # w.r.t. mean vector
            deltaCov_d = np.zeros((D, 1))  # w.r.t. covariance matrix
            for j in range(0, N):
                temp1 = np.divide(1, (sp[i].cov + sp[j].cov)) / 2
                temp2 = temp1 * (sp[i].mean - sp[j].mean)
                deltaMean_d = deltaMean_d + np.divide(temp2, 4)
                deltaCov_d = deltaCov_d + (temp1 - np.power(temp2, 2) / 4 - invCov_i) / 4

            # Calculate the Fisher information
            meanFisher = np.power(invCov_i, 2) * np.mean(np.power(difXtoMean, 2), 1).reshape(
                (-1, 1))  # w.r.t. mean vector
            covFisher = np.mean(
                np.tile(np.power(invCov_i, 2), (1, mu)) * np.power(difXtoMean, 2)
                - np.power(np.tile(invCov_i, reps=(1, mu)), 2),
                1) / 4  # w.r.t. covariance matrix

            # Update the probilistic model of the search process
            sp[i].mean = sp[i].mean + np.divide(1, meanFisher) * (
                    deltaMean_f + deltaMean_d * phi_init) * eta_m  # w.r.t. mean vector
            # w.r.t. covariance matrix
            sp[i].cov = sp[i].cov + np.divide(1, covFisher).reshape((-1, 1)) * (
                    deltaCov_f + deltaCov_d * phi_init) * eta_c

            # Boundary checking and repairing for mean vectors
            # problem != 7 and problem != 25:
            if filter:
                pos = sp[i].mean < lu[:, 0].reshape((-1, 1))
                sp[i].mean[pos] = 2 * lu[:, 0].reshape((-1, 1))[pos] - sp[i].mean[pos]
                pos = sp[i].mean > lu[:, 1].reshape((-1, 1))
                sp[i].mean[pos] = 2 * lu[:, 1].reshape((-1, 1))[pos] - sp[i].mean[pos]
                pos = sp[i].mean < lu[:, 0].reshape((-1, 1))
                sp[i].mean[pos] = lu[:, 0].reshape((-1, 1))[pos]

    outcome = min_f
    return outcome


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
