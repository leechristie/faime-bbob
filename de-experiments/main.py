import numpy as np
from bbobhelper import getbenchmark
from scipy.optimize import differential_evolution
import time
from datetime import datetime, timedelta

dim = 10

maxiter = 24
tol = 0
samples = 1000
popsize = 10


def logging_wrapper(func, flog):
    def rv(x):
        fitness = func(x)
        flog.append(fitness)
        return fitness
    rv.xmin = func.xmin
    rv.xmax = func.xmax
    rv.xopt = func.xopt
    rv.fopt = func.fopt
    return rv


def best_up_to_generation(log, evals_per_generation):
    rv = []
    assert len(log) % evals_per_generation == 0
    for i in range(len(log) // evals_per_generation):
        index = i * evals_per_generation
        rv.append(min(log[0:index+evals_per_generation]))
    return rv


def invert(x):
    major = len(x)
    minor = len(x[0])
    for i in range(major):
        assert len(x[i]) == minor
    rv = [[None] * major for i in range(minor)]
    for i in range(major):
        for j in range(minor):
            rv[j][i] = x[i][j]
    return rv


def print_lock(text):
    print(text, end="\n", flush=True)


start = time.time()
expected = len([0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34])\
           * len([0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94])\
           * samples
actual = 0


def print_progress_up():
    global expected
    global actual
    global start
    actual += 1
    elapsed = time.time() - start
    rate = elapsed / actual
    remaining = expected - actual
    rem_time = remaining * rate
    fin_time = datetime.now() + timedelta(seconds=rem_time)
    print("Progress : " + str(actual) + "/" + str(expected) +
          ", expected finish time : " + str(fin_time), end="\n", flush=True)


class myThread():
    def __init__(self, fid):
        self.fid = fid
    def run(self):
        fid = self.fid
        tasks = []
        index = 0
        for mutation in [0.25, 0.26, 0.27, 0.28, 0.29,
                         0.3, 0.31, 0.32, 0.33, 0.34]:
            for recombination in [0.85, 0.86, 0.87, 0.88, 0.89,
                                  0.9, 0.91, 0.92, 0.93, 0.94]:
                tasks.append(tuple([index, fid, mutation, recombination]))
                index += 1
        fidstr = str(fid)
        if fid < 10:
            fidstr = "0" + str(fid)
        with open("result/DEBUG_F" + fidstr + ".txt", "w") as debug:
            with open("result/output_F" + fidstr + ".csv", "w") as out:
                print("Function", "Mutation", "Recombination", "Generations",
                      "Meta Fitness (mean)", "Meta Fitness (stdev)", sep=",",
                      end="\n", file=out, flush=True)
                for task in tasks:
                    index, fid, mutation, recombination = task
                    summary_fitness_logs = []
                    for sample in range(samples):
                        fitness_log = []
                        doneCorrectly = False
                        while not doneCorrectly:
                            f = logging_wrapper(getbenchmark(fid, dim),
                                                fitness_log)
                            result = differential_evolution(f,
                                 [(f.xmin, f.xmax)] * dim,
                                 maxiter=maxiter, polish=False,
                                 tol=tol, recombination=recombination,
                                 mutation=mutation, popsize=popsize)
                            actual = len(fitness_log)
                            expected = popsize * dim * (maxiter+1)
                            if actual == expected:
                                doneCorrectly = True
                                summary_fitness_log \
                                    = best_up_to_generation(
                                    fitness_log, popsize * dim)
                                summary_fitness_logs.append(
                                    summary_fitness_log)
                                print_progress_up()
                            else:
                                print_lock("Sample failed. Calls to f : "
                                           + str(actual) + ", expected : "
                                           + str(expected))
                                print("Sample failed. Calls to f : "
                                      + str(actual) + ", expected : "
                                      + str(expected), file=debug,
                                      end="\n", flush=True)
                                fitness_log = []
                    inverted_summary_fitness_logs\
                        = invert(summary_fitness_logs)
                    generation_limit = 1
                    for fitnesses in inverted_summary_fitness_logs:
                        n = len(fitnesses)
                        mean = np.mean(fitnesses)
                        std = np.std(fitnesses, ddof=1)
                        print(fid, mutation, recombination, generation_limit,
                              mean, std, sep=",", end="\n", file=out,
                              flush=True)
                        generation_limit += 1


FID = int(input("FID: "))
t = myThread(FID)
t.run()

print("DONE.")
