import numpy as np
from bbobhelper import getbenchmark
from scipy.optimize import differential_evolution
import time
from datetime import datetime, timedelta

DIM = 10
MAX_ITER = 24
TOL = 0
SAMPLES = 1000
POP_SIZE = 10


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


MUTATION = [0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34]
RECOMBINATION = [0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94]


START = time.time()
EXPECTED = len(MUTATION) * len(RECOMBINATION) * SAMPLES
ACTUAL = 0


def print_progress_up():
    global EXPECTED
    global ACTUAL
    global START
    global DIM
    global MAX_ITER
    global TOL
    global SAMPLES
    global POP_SIZE
    ACTUAL += 1
    elapsed = time.time() - START
    rate = elapsed / ACTUAL
    remaining = EXPECTED - ACTUAL
    rem_time = remaining * rate
    fin_time = datetime.now() + timedelta(seconds=rem_time)
    print("Progress : " + str(ACTUAL) + "/" + str(EXPECTED) +
          ", expected finish time : " + str(fin_time), end="\n", flush=True)


def run_algorithm(index, fid, mutation, recombination, debug, out):
    global EXPECTED
    global ACTUAL
    global START
    global DIM
    global MAX_ITER
    global TOL
    global SAMPLES
    global POP_SIZE
    summary_fitness_logs = []
    for sample in range(SAMPLES):
        fitness_log = []
        done_correctly = False
        while not done_correctly:
            f = logging_wrapper(getbenchmark(fid, DIM),
                                fitness_log)
            differential_evolution(f, [(f.xmin, f.xmax)] * DIM,
                                   maxiter=MAX_ITER, polish=False,
                                   tol=TOL,
                                   recombination=recombination,
                                   mutation=mutation,
                                   popsize=POP_SIZE)
            ACTUAL = len(fitness_log)
            EXPECTED = POP_SIZE * DIM * (MAX_ITER + 1)
            if ACTUAL == EXPECTED:
                done_correctly = True
                summary_fitness_log \
                    = best_up_to_generation(
                    fitness_log, POP_SIZE * DIM)
                summary_fitness_logs.append(
                    summary_fitness_log)
                print_progress_up()
            else:
                print("Sample failed. Calls to f : " + str(ACTUAL)
                      + ", expected : " + str(EXPECTED))
                print("Sample failed. Calls to f : " + str(ACTUAL)
                      + ", expected : " + str(EXPECTED), file=debug,
                      end="\n", flush=True)
                fitness_log = []
    inverted_summary_fitness_logs \
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


def run_experiment(fid):
    global EXPECTED
    global ACTUAL
    global START
    global MUTATION
    global RECOMBINATION
    global DIM
    global MAX_ITER
    global TOL
    global SAMPLES
    global POP_SIZE
    tasks = []
    index = 0
    for mutation in MUTATION:
        for recombination in RECOMBINATION:
            tasks.append((index, fid, mutation, recombination))
            index += 1
    fid_str = str(fid)
    if fid < 10:
        fid_str = "0" + str(fid)
    with open("result/DEBUG_F" + fid_str + ".txt", "w") as debug:
        with open("result/output_F" + fid_str + ".csv", "w") as out:
            print("Function", "Mutation", "Recombination", "Generations",
                  "Meta Fitness (mean)", "Meta Fitness (stdev)", sep=",",
                  end="\n", file=out, flush=True)
            for task in tasks:
                index, fid, mutation, recombination = task
                run_algorithm(index, fid, mutation, recombination, debug, out)


def main():
    fid = int(input("fid: "))
    run_experiment(fid)
    print("DONE.")


if __name__ == '__main__':
    main()
