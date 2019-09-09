from collections import namedtuple

import numpy as np

from algorithm.other_ncs import NCS
from algorithm.benchmark import benchmark_func
from algorithm.problem import load_problem

SuperNCSParameter = namedtuple("SuperNCSParameter",
                               ['init_value',
                                'stepsize',
                                'bounds',
                                'ftarget',
                                'popsize',
                                "tmax",
                                "reset_xl_to_pop",
                                "init_pop",
                                "best_k",
                                ])



if __name__ == '__main__':
    problem_set = [6, 12]
    for p in problem_set:
        print("\n************ the problem %d started! ************\n" % p)
        # __C.parameters = {'reset_xl_to_pop': False, 'init_value': tmp_crates, 'stepsize': ncs_stepsize,
        #                   'bounds': [0.0, 10.], 'ftarget': 0, 'tmax': 1600, 'popsize': 8, 'best_k': 1}
        D = 30
        parameters = load_problem(p, D)
        ncs_para = SuperNCSParameter(reset_xl_to_pop=False, init_value=np.ones(D),
                                     stepsize=50,bounds=parameters.lu[0],
                                     ftarget=0, tmax=100000, popsize=10, best_k=1, init_pop=None)
        fitness = benchmark_func(ncs_para.init_value.reshape((-1, 1)), p, parameters)
        sigma = [abs(ncs_para.bounds[0] - ncs_para.bounds[1])/10] * D
        es = NCS(ncs_para)
        es.set_initFitness(fitness=np.tile(fitness, reps=(ncs_para.popsize)), sigma=sigma, r=0.99)
        es.epoch = 10
        # evolution loop
        while not es.stop():
            x = es.ask()
            fit = benchmark_func(np.asarray(x).T, p, parameters)
            # print X,fit
            es.tell(x, fit)
            es.disp(1000)

        print('the {} th problem result is:'.format(p))
        # print('the mean result is: {} and the std is {}'.format((np.mean(outcome)), np.std(outcome)))


