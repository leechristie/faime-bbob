#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Process data and generates some comparison results.

   Synopsis:
      python path_to_folder/bbob_pproc/runcompall.py [OPTIONS] FOLDER_NAME...

   Help:
      python path_to_folder/bbob_pproc/runcompall.py -h

"""



import os, sys
import warnings
import numpy
from pdb import set_trace

# Add the path to bbob_pproc
if __name__ == "__main__":
    # append path without trailing '/bbob_pproc', using os.sep fails in mingw32
    #sys.path.append(filepath.replace('\\', '/').rsplit('/', 1)[0])
    (filepath, filename) = os.path.split(sys.argv[0])
    #Test system independent method:
    sys.path.append(os.path.join(filepath, os.path.pardir))
    import matplotlib
    matplotlib.use('Agg') # To avoid window popup and use without X forwarding

from bbob_pproc import genericsettings, pproc
from bbob_pproc.pproc import DataSetList
from bbob_pproc.ppfig import saveFigure
from bbob_pproc.toolsstats import prctile
    
import matplotlib.pyplot as plt

warned = False  # print just one warning and set to True

#FUNCTION DEFINITIONS


def rearrange(blist, flist):
    """Alligns the number of evaluations taken from the blist with the correpsning flist"""
    final_b=[]
    final_f=[]
    for i in range(0,len(blist)): #runs over dimensions
        erg_b = numpy.empty((0), float)
        erg_f = [numpy.empty ((0), float), numpy.empty ((0), float), numpy.empty ((0), float)]
        for j in range(0,len(blist[i])): #runs over function evaluations
            erg_b=numpy.append(erg_b,blist[i][j])
            erg_f[0]=numpy.append(erg_f[0],numpy.median(flist[i][j]))
            erg_f[1]=numpy.append(erg_f[1],prctile(flist[i][j], [0.25]))
            erg_f[2]=numpy.append(erg_f[2],prctile(flist[i][j], [0.75]))
        final_b.append(erg_b)
        final_f.append(erg_f)
    return final_b, final_f

def main(dictAlg, outputdir='.', verbose=True):
    """Main routine for generating convergence plots

    """
    global warned  # bind variable warned into this scope
    dictFun = pproc.dictAlgByFun(dictAlg)
    for l in dictFun:  # l appears to be the function id!?
        for i in dictFun[l]: # please, what is i??? appears to be the algorithm-key
            plt.figure()
            if type(i) in (list, tuple):
                figurename = "ppconv_plot_" + i[0] + "_f" + str(l)
            else:
                try:
                    figurename = "ppconv_plot_" + dictFun[l][i].algId + "_f" + str(l)
                except AttributeError:  # this is a (rather desperate) bug-fix attempt that works for the unit test
                    figurename = "ppconv_plot_" + dictFun[l][i][0].algId + "_f" + str(l)
            plt.xlabel('number of function evaluations / dimension')
            plt.ylabel('Median of fitness')
            plt.grid()
            ax = plt.gca()
            ax.set_yscale("log")
            ax.set_xscale("log")
            for j in dictFun[l][i]: # please, what is j??? a dataset
                dimList_b = []
                dimList_f = []
                dimList_b.append(j.funvals[:,0])
                dimList_f.append(j.funvals[:,1:])
                bs, fs= rearrange(dimList_b, dimList_f)
                labeltext=str(j.dim)+"D"
                try:
                    if 11 < 3:
                        plt.errorbar(bs[0] / j.dim, fs[0][0], yerr = [fs[0][1], fs[0][2]], label = labeltext)
                    else:
                        plt.errorbar(bs[0] / j.dim, fs[0][0], label = labeltext)
                except FloatingPointError:  # that's a bit of a hack
                    if not warned:
                        print('Warning: floating point error when plotting errorbars, ignored')
                    warned = True
            plt.legend(loc=3)
            saveFigure(os.path.join(outputdir, figurename.replace(' ','')),  genericsettings.fig_formats, verbose=verbose)
            plt.close()
    print("Convergence plots done.")
        
if __name__ == "__main__":
    sys.exit(main())
