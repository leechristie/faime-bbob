#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module for post-processing the data of one algorithm.

Calls the function main with arguments from the command line. Executes
the postprocessing on the given files and folders arguments, using the
:file:`.info` files found recursively.

Synopsis:
    ``python path_to_folder/bbob_pproc/rungeneric1.py [OPTIONS] FOLDER``

Help:
    ``python path_to_folder/bbob_pproc/rungeneric1.py -h``

"""



import os, sys, warnings, getopt, numpy as np
from pdb import set_trace

xrange = range

# Add the path to bbob_pproc
if __name__ == "__main__":
    # os.path.split is system independent
    (filepath, filename) = os.path.split(sys.argv[0])
    sys.path.append(os.path.join(filepath, os.path.pardir))
    import matplotlib
    matplotlib.use('Agg')  # To avoid window popup and use without X forwarding

from bbob_pproc import genericsettings, pptable, pprldistr, ppfigdim, pplogloss, findfiles
from bbob_pproc.pproc import DataSetList
from bbob_pproc.toolsdivers import print_done, prepend_to_file, strip_pathname, str_to_latex
from bbob_pproc import ppconverrorbars

import matplotlib.pyplot as plt

__all__ = ['main']

# Used by getopt:
shortoptlist = "hvpfo:"
longoptlist = ["help", "output-dir=", "noisy", "noise-free", "tab-only",
               "fig-only", "rld-only", "los-only", "crafting-effort=",
               "pickle", "verbose", "settings=", "conv", "expensive", 
               "not-expensive", "runlength-based"]

# CLASS DEFINITIONS

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


# FUNCTION DEFINITIONS

def usage():
    print(main.__doc__)

def main(argv=None):
    r"""Post-processing COCO data of a single algorithm.

    Provided with some data, this routine outputs figure and TeX files
    in a folder needed for the compilation of latex document
    :file:`template1XXX.tex` or :file:`noisytemplate1XXX.tex`, where 
    :file:`XXX` is either :file:`ecj` or :file:`generic`. The template
    file needs to be edited so that the commands
    ``\bbobdatapath`` and ``\algfolder`` point to the output folder.

    These output files will contain performance tables, performance
    scaling figures and empirical cumulative distribution figures. On
    subsequent executions, new files will be added to the output folder,
    overwriting existing older files in the process.

    Keyword arguments:

    *argv* -- list of strings containing options and arguments. If not
    given, sys.argv is accessed.

    *argv* should list either names of :file:`info` files or folders
    containing :file:`info` files. argv can also contain post-processed
    :file:`pickle` files generated by this routine. Furthermore, *argv*
    can begin with, in any order, facultative option flags listed below.

        -h, --help
            displays this message.
        -v, --verbose
            verbose mode, prints out all operations.
        -p, --pickle
            generates pickle post processed data files.
        -o OUTPUTDIR, --output-dir=OUTPUTDIR
            changes the default output directory (:file:`ppdata`) to
            :file:`OUTPUTDIR`.
        --crafting-effort=VALUE
            sets the crafting effort to VALUE (float). Otherwise the
            default value of 0. will be used.
        --noise-free, --noisy
            processes only part of the data.
        --settings=SETTINGS
            changes the style of the output figures and tables. At the
            moment the only differences are  in the colors of the output
            figures. SETTINGS can be either "grayscale", "color" or
            "black-white". The default setting is "color".
        --tab-only, --fig-only, --rld-only, --los-only
            these options can be used to output respectively the TeX
            tables, convergence and ERTs graphs figures, run length
            distribution figures, ERT loss ratio figures only. A
            combination of any two of these options results in no
            output.
        --conv
            if this option is chosen, additionally convergence plots
            for each function and algorithm are generated.
        --expensive
            runlength-based f-target values and fixed display limits,
            useful with comparatively small budgets. By default the
            setting is based on the budget used in the data.
        --not-expensive
            expensive setting off. 
        --runlength-based
            runlength-based f-target values, such that the
            "level of difficulty" is similar for all functions. 

    Exceptions raised:

    *Usage* -- Gives back a usage message.

    Examples:

    * Calling the rungeneric1.py interface from the command line::

        $ python bbob_pproc/rungeneric1.py -v experiment1

      will post-process the folder experiment1 and all its containing
      data, base on the .info files found in the folder. The result will
      appear in the default output folder. The -v option adds verbosity. ::

        $ python bbob_pproc/rungeneric1.py -o exp2 experiment2/*.info

      This will execute the post-processing on the info files found in
      :file:`experiment2`. The result will be located in the alternative
      location :file:`exp2`.

    * Loading this package and calling the main from the command line
      (requires that the path to this package is in python search path)::

        $ python -m bbob_pproc.rungeneric1 -h

      This will print out this help message.

    * From the python interpreter (requires that the path to this
      package is in python search path)::

        >> import bbob_pproc as bb
        >> bb.rungeneric1.main('-o outputfolder folder1'.split())

      This will execute the post-processing on the index files found in
      :file:`folder1`. The ``-o`` option changes the output folder from
      the default to :file:`outputfolder`.

    """

    if argv is None:
        argv = sys.argv[1:]
        # The zero-th input argument which is the name of the calling script is
        # disregarded.

    if 1 < 3:
        opts, args = getopt.getopt(argv, shortoptlist, longoptlist)
        if 11 < 3:
            try:
                opts, args = getopt.getopt(argv, shortoptlist, longoptlist)
            except getopt.error as msg:
                raise Usage(msg)

        if not (args) and not '--help' in argv and not 'h' in argv:
            print('not enough input arguments given')
            print('cave: the following options also need an argument:')
            print([o for o in longoptlist if o[-1] == '='])
            print('options given:')
            print(opts)
            print('try --help for help')
            sys.exit()

        inputCrE = 0.
        isfigure = True
        istab = True
        isrldistr = True
        islogloss = True
        isPostProcessed = False
        isPickled = False
        verbose = False
        outputdir = 'ppdata'
        isNoisy = False
        isNoiseFree = False
        inputsettings = 'color'
        isConv = False
        isRLbased = None  # allows automatic choice
        isExpensive = None 

        # Process options
        for o, a in opts:
            if o in ("-v", "--verbose"):
                verbose = True
            elif o in ("-h", "--help"):
                usage()
                sys.exit()
            elif o in ("-p", "--pickle"):
                isPickled = True
            elif o in ("-o", "--output-dir"):
                outputdir = a
            elif o == "--noisy":
                isNoisy = True
            elif o == "--noise-free":
                isNoiseFree = True
            # The next 4 are for testing purpose
            elif o == "--tab-only":
                isfigure = False
                isrldistr = False
                islogloss = False
            elif o == "--fig-only":
                istab = False
                isrldistr = False
                islogloss = False
            elif o == "--rld-only":
                istab = False
                isfigure = False
                islogloss = False
            elif o == "--los-only":
                istab = False
                isfigure = False
                isrldistr = False
            elif o == "--crafting-effort":
                try:
                    inputCrE = float(a)
                except ValueError:
                    raise Usage('Expect a valid float for flag crafting-effort.')
            elif o == "--settings":
                inputsettings = a
            elif o == "--conv":
                isConv = True
            elif o == "--runlength-based":
                isRLbased = True
            elif o == "--expensive":
                isExpensive = True  # comprises runlength-based
            elif o == "--not-expensive":
                isExpensive = False  
            else:
                assert False, "unhandled option"

        # from bbob_pproc import bbob2010 as inset # input settings
        if inputsettings == "color":
            from bbob_pproc import genericsettings as inset  # input settings
        elif inputsettings == "grayscale":
            from bbob_pproc import grayscalesettings as inset  # input settings
        elif inputsettings == "black-white":
            from bbob_pproc import bwsettings as inset  # input settings
        else:
            txt = ('Settings: %s is not an appropriate ' % inputsettings
                   + 'argument for input flag "--settings".')
            raise Usage(txt)
        
        if 11 < 3:
            from bbob_pproc import config  # input settings
            config.config()
            import imp
            # import testbedsettings as testbedsettings # input settings
            try:
                fp, pathname, description = imp.find_module("testbedsettings")
                testbedsettings = imp.load_module("testbedsettings", fp, pathname, description)
            finally:
                fp.close()

        if (not verbose):
            warnings.simplefilter('module')
            # warnings.simplefilter('ignore')            

        print(("Post-processing (1): will generate output " + 
               "data in folder %s" % outputdir))
        print("  this might take several minutes.")

        filelist = list()
        for i in args:
            i = i.strip()
            if os.path.isdir(i):
                filelist.extend(findfiles.main(i, verbose))
            elif os.path.isfile(i):
                filelist.append(i)
            else:
                txt = 'Input file or folder %s could not be found.' % i
                print(txt)
                raise Usage(txt)
        dsList = DataSetList(filelist, verbose)
        
        if not dsList:
            raise Usage("Nothing to do: post-processing stopped.")

        if isNoisy and not isNoiseFree:
            dsList = dsList.dictByNoise().get('nzall', DataSetList())
        if isNoiseFree and not isNoisy:
            dsList = dsList.dictByNoise().get('noiselessall', DataSetList())

        # compute maxfuneval values
        dict_max_fun_evals = {}
        for ds in dsList:
            dict_max_fun_evals[ds.dim] = np.max((dict_max_fun_evals.setdefault(ds.dim, 0), float(np.max(ds.maxevals))))
        if isRLbased is not None:
            genericsettings.runlength_based_targets = isRLbased

        from bbob_pproc import config
        config.target_values(isExpensive, dict_max_fun_evals)
        config.config()

        if (verbose):
            for i in dsList:
                if (dict((j, i.instancenumbers.count(j)) for j in set(i.instancenumbers)) != 
                    inset.instancesOfInterest):
                    warnings.warn('The data of %s do not list ' % (i) + 
                                  'the correct instances ' + 
                                  'of function F%d.' % (i.funcId))

        dictAlg = dsList.dictByAlg()

        if len(dictAlg) > 1:
            warnings.warn('Data with multiple algId %s ' % (dictAlg) + 
                          'will be processed together.')
            # TODO: in this case, all is well as long as for a given problem
            # (given dimension and function) there is a single instance of
            # DataSet associated. If there are more than one, the first one only
            # will be considered... which is probably not what one would expect.
            # TODO: put some errors where this case would be a problem.
            # raise Usage?

        if isfigure or istab or isrldistr or islogloss:
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
                if verbose:
                    print('Folder %s was created.' % (outputdir))

        if isPickled:
            dsList.pickle(verbose=verbose)

        if isConv:
            ppconverrorbars.main(dictAlg, outputdir, verbose)

        if isfigure:
            print("Scaling figures...", end=' ')
            sys.stdout.flush()
            # ERT/dim vs dim.
            plt.rc("axes", **inset.rcaxeslarger)
            plt.rc("xtick", **inset.rcticklarger)
            plt.rc("ytick", **inset.rcticklarger)
            plt.rc("font", **inset.rcfontlarger)
            plt.rc("legend", **inset.rclegendlarger)
            ppfigdim.main(dsList, ppfigdim.values_of_interest, outputdir, verbose)
            plt.rcdefaults()
            print_done()

        plt.rc("axes", **inset.rcaxes)
        plt.rc("xtick", **inset.rctick)
        plt.rc("ytick", **inset.rctick)
        plt.rc("font", **inset.rcfont)
        plt.rc("legend", **inset.rclegend)

        if istab:
            print("TeX tables...", end=' ')
            sys.stdout.flush()
            dictNoise = dsList.dictByNoise()
            for noise, sliceNoise in dictNoise.items():
                pptable.main(sliceNoise, inset.tabDimsOfInterest, outputdir,
                             noise, verbose)
            print_done()

        if isrldistr:
            print("ECDF graphs...", end=' ')
            sys.stdout.flush()
            dictNoise = dsList.dictByNoise()
            if len(dictNoise) > 1:
                warnings.warn('Data for functions from both the noisy and '
                              'non-noisy testbeds have been found. Their '
                              'results will be mixed in the "all functions" '
                              'ECDF figures.')
            dictDim = dsList.dictByDim()
            for dim in inset.rldDimsOfInterest:
                try:
                    sliceDim = dictDim[dim]
                except KeyError:
                    continue

                pprldistr.main(sliceDim, True,
                               outputdir, 'all', verbose)
                dictNoise = sliceDim.dictByNoise()
                for noise, sliceNoise in dictNoise.items():
                    pprldistr.main(sliceNoise, True,
                                   outputdir, '%s' % noise, verbose)
                dictFG = sliceDim.dictByFuncGroup()
                for fGroup, sliceFuncGroup in list(dictFG.items()):
                    pprldistr.main(sliceFuncGroup,
                                   True, outputdir, '%s' % fGroup, verbose)

                pprldistr.fmax = None  # Resetting the max final value
                pprldistr.evalfmax = None  # Resetting the max #fevalsfactor

            print_done()

        if islogloss:
            print("ERT loss ratio figures and tables...", end=' ')
            sys.stdout.flush()
            for ng, sliceNoise in dsList.dictByNoise().items():
                if ng == 'noiselessall':
                    testbed = 'noiseless'
                elif ng == 'nzall':
                    testbed = 'noisy'
                txt = ("Please input crafting effort value "
                       + "for %s testbed:\n  CrE = " % testbed)
                CrE = inputCrE
                while CrE is None:
                    try:
                        CrE = float(input(txt))
                    except (SyntaxError, NameError, ValueError):
                        print("Float value required.")
                dictDim = sliceNoise.dictByDim()
                for d in inset.rldDimsOfInterest:
                    try:
                        sliceDim = dictDim[d]
                    except KeyError:
                        continue
                    info = '%s' % ng
                    pplogloss.main(sliceDim, CrE, True, outputdir, info,
                                   verbose=verbose)
                    pplogloss.generateTable(sliceDim, CrE, outputdir, info,
                                            verbose=verbose)
                    for fGroup, sliceFuncGroup in sliceDim.dictByFuncGroup().items():
                        info = '%s' % fGroup
                        pplogloss.main(sliceFuncGroup, CrE, True, outputdir, info,
                                       verbose=verbose)
                    pplogloss.evalfmax = None  # Resetting the max #fevalsfactor

            print_done()

        latex_commands_file = os.path.join(outputdir.split(os.sep)[0], 'bbob_pproc_commands.tex')
        prepend_to_file(latex_commands_file,
                        ['\\providecommand{\\bbobloglosstablecaption}[1]{', 
                         pplogloss.table_caption, '}'])
        prepend_to_file(latex_commands_file,
                        ['\\providecommand{\\bbobloglossfigurecaption}[1]{', 
                         pplogloss.figure_caption, '}'])
        prepend_to_file(latex_commands_file,
                        ['\\providecommand{\\bbobpprldistrlegend}[1]{',
                         pprldistr.caption_single(np.max([ val / dim for dim, val in dict_max_fun_evals.items()])),  # depends on the config setting, should depend on maxfevals
                         '}'])
        prepend_to_file(latex_commands_file,
                        ['\\providecommand{\\bbobppfigdimlegend}[1]{',
                         ppfigdim.scaling_figure_caption(),
                         '}'])
        prepend_to_file(latex_commands_file,
                        ['\\providecommand{\\bbobpptablecaption}[1]{',
                         pptable.table_caption,
                         '}'])
        prepend_to_file(latex_commands_file,
                        ['\\providecommand{\\algfolder}{}'])  # is overwritten in rungeneric.py
        prepend_to_file(latex_commands_file,
                        ['\\providecommand{\\algname}{' + 
                         (str_to_latex(strip_pathname(args[0])) if len(args) == 1 else str_to_latex(dsList[0].algId)) + '{}}'])
        if isfigure or istab or isrldistr or islogloss:
            print("Output data written to folder %s" % outputdir)

        plt.rcdefaults()

#    except Usage, err:
#        print >> sys.stderr, err.msg
#        print >> sys.stderr, "for help use -h or --help"
#        return 2


if __name__ == "__main__":
    res = main()
    if genericsettings.test: 
        print(res)
    sys.exit(res)
