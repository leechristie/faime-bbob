from bbob3 import bbobbenchmarks
import numpy as np


def getbenchmark(fid, dim, instance=None, zerox=False, zerof=True, param=None):
    """Returns an instance of the specified BBOB function
    
    Keyword arguments:
    fid -- the funciton ID (1 to 24)
    dim -- the number of dimensions (positive)
    instance -- the instance seed (default is randomly generated)
    zerox -- center fopt at the zero vector if possible (default False)
    zerof -- global optimum fitness is 0 (default True)
    param -- funtion specific parameter (default None)
    """
    assert type(fid) == int
    assert 1 <= fid <= 24
    assert type(dim) == int
    assert dim >= 1
    assert instance == None or type(instance) == int
    assert type(zerox) == bool
    assert type(zerof) == bool
    if instance == None:
        instance = np.random.randint(low=-2147483648, high=2147483647) + 2147483648
    benchmark = getattr(bbobbenchmarks, "F" + str(fid))
    f = benchmark(instance, zerox=zerox, zerof=zerof, param=param)
    f.initwithsize((dim,), dim)
    assert f.dim == dim
    f.xmin = -5
    f.xmax = 5
    f.maximising = False
    f.__name__ = f.shortstr()
    f.fid = fid
    return f


if __name__ == '__main__':
    DIM = 4
    for i in range(1, 24+1):
        f = getbenchmark(i, DIM)
        print(f.__name__, "with", f.fopt, "at", f.xopt)
