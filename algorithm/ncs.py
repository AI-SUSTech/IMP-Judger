import math
import numpy as np
import time


def load_problem(problem_path, dim):
    print("load problem in ", problem_path)

    (o, A, M, a, alpha, b, lu) = [None] * 7
    problem6_opti = "../datasets_ncs/rosenbrock_func_data.txt"
    with open(problem6_opti, 'r') as f:
        data = f.readlines()[0].split(" ")
        optimal = []
        i = 0
        for d in data:
            if d != "":
                optimal.append(float(d))
                i += 1
        o = np.asarray(optimal)
    b = 390 # just for problem 6, see fbias_data.txt
    o = o[0:dim]
    lu = np.asarray([[-100, 100]]* dim)  # D X 2
    return o, A, M, a, alpha, b, lu


class Struct:
    def __init__(self, x, fit, mean, cov):
        self.x = x
        self.fit = fit
        self.mean = mean
        self.cov = cov

    def __iter__(self):
        pass


def repmat(e: Struct, shape):
    return np.full(e, shape)


def benchmark_func(x, problem, o, A, M, a, alpha, b):
    """
    :param x: the solution that to be judged
    :param problem: problem index
    :param o: the optimal solution
    :param A:
    :param M:
    :param a:
    :param alpha:
    :param b: f_bias
    :return:
    """
    # only support problem 6
    dimension, num_sol = x.shape
    fitness = np.zeros(1, num_sol)
    for i in range(num_sol):
        onefitness = b
        z = x - o + 1
        for d in range(dimension-1):
            onefitness += 100*(z[i]**2-z[i+1])**2 + (z[i] - 1)**2
        fitness[i] = onefitness
    return fitness


def ncs(problem_index, filter=True):
    # pre load the data from the problem
    D = 30  # the dimension of the problem
    o, A, M, a, alpha, b, lu = load_problem(problem_index, D)

    # Configuration of NCS
    _lambda = math.ceil(np.log(D))  # the number of search processes
    mu = 4 + math.floor(3 * np.log(D))  # the number of solutions in each search process
    phi_init = 0.00005  # the trade-off parameter between f and d
    eta_m_init = 1  # the step size of gradient descent for mean vector
    eta_c_init = (3 + np.log(D)) / (5 * np.sqrt(D))  # the step size of gradient descent for mean vector
    #  todo this function is to be validate, not the same as repmat in matlab
    vl = np.tile(lu[:, 0], (1, mu))
    vu = np.tile(lu[:, 1], (1, mu))

    # Configuration of the test protocol
    MAXFES = 10000 * D  # the total FE of each run
    total_time = 25  # the total number of runs

    # Record the best results for each problem
    outcome = np.multiply(np.ones(total_time, 1), 1e300)

    # Definition of the structure of search processes
    sp = repmat(
        Struct(x=np.zeros(D, mu), fit=np.zeros(1, mu), mean=np.zeros(D, 1), cov=np.zeros(D, 1)),
        (_lambda, 1)
    )
    current_time = 1

    while current_time <= total_time:

        # Set the randomness seed
        np.random.seed(time.time())

        # Re-initialize the best solution recorder in this run
        min_f = 1e300
        FES = 0

        for i in range(1, _lambda):
            # Model the search process as Gaussian probabilistic distribution
            sp[i].mean = lu[:, 0] + np.multiply(
                np.random.rand((D, 1)),
                (lu[:, 1] - lu[:, 0]))

            sp[i].cov = np.divide((lu[:, 1] - lu[:, 0]), _lambda)

        # The main loop body
        while FES < MAXFES:

            eta_m = np.multiply(eta_m_init, ((math.exp(1) - math.exp(FES / MAXFES)) / (math.exp(1) - 1)))
            eta_c = np.multiply(eta_c_init, ((math.exp(1) - math.exp(FES / MAXFES)) / (math.exp(1) - 1)))

            for i in range(1, _lambda):
                # Generate mu solutions for each search process
                sp[i].x = np.tile(sp[i].mean, (1, mu)) + \
                          np.multiply(
                              np.random.normal(size=(D, mu)),
                              (np.tile(sp[i].cov, (1, mu)))
                          )
                # Boundary checking and repairing
                # problem != 7 and problem != 25:
                if filter:
                    # todo < in matlab is different in python
                    pos = sp[i].x < vl
                    sp[i].x[pos] = np.multiply(2, vl[pos]) - sp[i].x[pos]
                    pos = sp[i].x > vu
                    sp[i].x[pos] = np.multiply(2, vu[pos]) - sp[i].x[pos]
                    pos = sp[i].x < vl
                    sp[i].x[pos] = vl[pos]

                    # Fitness evalution for mu solutions
                sp[i].fit = benchmark_func(sp[i].x, problem_index, o, A, M, a, alpha, b)
                FES = FES + mu

                # Update the best solution ever found
                min_f = min(min(sp[i].fit), min_f)

                # Rank mu solutions ascendingly in terms of fitness
                order = np.sort(sp[i].fit)
                rank = np.sort(order)

                # Set utility value for mu solutions in terms of rank
                tempU = max(0, np.log(mu / 2 + 1) - np.log(rank.T))
                utility = np.divide(tempU, sum(tempU)) - 1 / mu

                # Prepare for calculating gradients (for saving computation current_time)
                invCov_i = np.divide(1, sp[i].cov)
                difXtoMean = sp[i].x - np.tile(sp[i].mean, reps=(1, mu))

                # Calculate the gradients of expectation of fitness values    
                deltaMean_f = np.multiply(invCov_i,
                                          np.mean(
                                              np.multiply(
                                                  difXtoMean,
                                                  np.tile(utility, reps=(D, 1))),
                                              2))  # w.r.t. mean vector
                deltaCov_f = np.divide(np.multiply(np.power(invCov_i, 2),
                                                   np.mean(np.multiply(np.power(difXtoMean, 2),
                                                                       np.tile(utility, reps=(D, 1))),
                                                           2)),
                                       2)  # w.r.t. covariance matrix
                # Calculate the gradients of distribution distances                
                deltaMean_d = np.zeros((D, 1))  # w.r.t. mean vector
                deltaCov_d = np.zeros((D, 1))  # w.r.t. covariance matrix
                for j in range(1, _lambda):
                    temp1 = np.divide(np.divide(1, (sp[i].cov + sp[j].cov)), 2)
                    temp2 = np.multiply(temp1, (sp[i].mean - sp[j].mean))
                    deltaMean_d = deltaMean_d + np.divide(temp2, 4)
                    deltaCov_d = deltaCov_d + np.divide((temp1 - np.divide(np.power(temp2, 2), 4) - invCov_i), 4)

                # Calculate the Fisher information 
                meanFisher = np.multiply(np.power(invCov_i, 2),
                                         np.mean(np.power(difXtoMean, 2), 2))  # w.r.t. mean vector
                covFisher = np.divide(
                    np.mean(
                        np.power(
                            (np.power(
                                np.multiply(
                                    np.tile(np.power(invCov_i, 2), 1, mu),
                                    difXtoMean
                                ), 2) - np.tile(invCov_i, reps=(1, mu))),
                            2),
                        2),
                    4)  # w.r.t. covariance matrix

                # Update the probilistic model of the search process
                sp[i].mean = np.multiply(np.multiply(sp[i].mean + np.divide(1, meanFisher),
                                                     (deltaMean_f + np.multiply(deltaMean_d, phi_init))),
                                         eta_m)  # w.r.t. mean vector
                sp[i].cov = np.multiply(
                    np.multiply(sp[i].cov + np.divide(1, covFisher), (deltaCov_f + np.multiply(deltaCov_d, phi_init))),
                    eta_c)  # w.r.t. covariance matrix

                # Boundary checking and repairing for mean vectors
                # problem != 7 and problem != 25:
                if filter:
                    # todo < in matlab is different in python
                    pos = sp[i].mean < lu[:, 1]
                    sp[i].mean[pos] = np.multiply(2, lu[pos, 1]) - sp[i].mean[pos]
                    pos = sp[i].mean > lu[:, 2]
                    sp[i].mean[pos] = np.multiply(2, lu[pos, 2]) - sp[i].mean[pos]
                    pos = sp[i].mean < lu[:, 1]
                    sp[i].mean[pos] = lu[pos, 1]

                    # Print the best solution ever found to the screen
            print('The best result at the {} th FE is {}'.format(FES, min_f))
        outcome[current_time] = min_f
        current_time = current_time + 1
    return outcome


if __name__ == '__main__':
    problem_set = list(range(6, 7))
    for p in problem_set:
        print("************the problem %d started!************" % p)

        if p == 7 or p == 25:
            outcome = ncs(p, False)
        else:
            outcome = ncs(p)

        print('the {} th problem result is:'.format(p))
        print('the mean result is: {} and the std is {}'.format((np.mean(outcome)), np.std(outcome)))
