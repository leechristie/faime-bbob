#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Process data to be included in a latex template. 

Synopsis:
    ``python path_to_folder/bbob_pproc/rungeneric.py [OPTIONS] FOLDERS``

Help:
    ``python path_to_folder/bbob_pproc/rungeneric.py -h``

"""



import os
import sys
import glob
import getopt
import pickle
import tarfile
from pdb import set_trace
import warnings
import numpy
# numpy.seterr(all='raise')

xrange = range

# Add the path to bbob_pproc
if __name__ == "__main__":
    (filepath, filename) = os.path.split(sys.argv[0])
    #Test system independent method:
    sys.path.append(os.path.join(filepath, os.path.pardir))
    import matplotlib
    matplotlib.use('Agg') # To avoid window popup and use without X forwarding

from bbob_pproc import genericsettings, rungeneric1, rungeneric2, rungenericmany
from bbob_pproc.toolsdivers import prepend_to_file, truncate_latex_command_file, print_done

__all__ = ['main']

# Combine optlist for getopt:
# Make a set of the short option list, has one-letter elements that could be
# followed by colon
shortoptlist = set()
for i in (rungeneric1.shortoptlist, rungeneric2.shortoptlist,
          rungenericmany.shortoptlist):
    tmp = i[:]
    # split into logical elements: one-letter that could be followed by colon
    while tmp:
        if len(tmp) > 1 and tmp[1] is ':':
            shortoptlist.add(tmp[0:2])
            tmp = tmp[2:]
        else:
            shortoptlist.add(tmp[0])
            tmp = tmp[1:]
shortoptlist = ''.join(shortoptlist)

longoptlist = list(set.union(set(rungeneric1.longoptlist),
                             set(rungeneric2.longoptlist),
                             set(rungenericmany.longoptlist)))

#CLASS DEFINITIONS

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

#FUNCTION DEFINITIONS

def _splitshortoptlist(shortoptlist):
    """Split short options list used by getopt.

    Returns a set of the options.

    """
    res = set()
    tmp = shortoptlist[:]
    # split into logical elements: one-letter that could be followed by colon
    while tmp:
        if len(tmp) > 1 and tmp[1] is ':':
            res.add(tmp[0:2])
            tmp = tmp[2:]
        else:
            res.add(tmp[0])
            tmp = tmp[1:]

    return res

def usage():
    print(main.__doc__)

def main(argv=None):
    r"""Main routine for post-processing data from COCO.

    Depending on the number of data path input arguments, this routine will:

    * call sub-routine :py:func:`bbob_pproc.rungeneric1.main` for each
      input arguments; each input argument will be used as output
      sub-folder relative to the main output folder,
    * call either sub-routines :py:func:`bbob_pproc.rungeneric2.main`
      (2 input arguments) or :py:func:`bbob_pproc.rungenericmany.main`
      (more than 2) for the input arguments altogether.

    The output figures and tables written by default to the output folder 
    :file:`ppdata` are used in latex templates:

    * :file:`template1generic.tex`, :file:`noisytemplate1generic.tex`
      for results with a **single** algorithm on the noise-free and noisy
      testbeds respectively
    * :file:`template2generic.tex`, :file:`noisytemplate2generic.tex`, 
      for showing the comparison of **2** algorithms
    * :file:`template3generic.tex`, :file:`noisytemplate3generic.tex` 
      for showing the comparison of **more than 2** algorithms.

    These latex templates need to be copied in the current working directory 
    and possibly edited so that the LaTeX commands ``\bbobdatapath`` and
    ``\algfolder`` point to the correct output folders of the post-processing. 
    Compiling the template file with LaTeX should then produce a document.

    Keyword arguments:

    *argv* -- list of strings containing options and arguments. If not
       provided, sys.argv is accessed.

    *argv* must list folders containing COCO data files. Each of these
    folders should correspond to the data of ONE algorithm.

    Furthermore, argv can begin with facultative option flags.

        -h, --help

            displays this message.

        -v, --verbose

            verbose mode, prints out operations.

        -o, --output-dir=OUTPUTDIR

            changes the default output directory (:file:`ppdata`) to
            :file:`OUTPUTDIR`.
        
        --omit-single
        
            omit calling :py:func:`bbob_pproc.rungeneric1.main`, if
            more than one data path argument is provided. 
            
        --rld-single-fcts
        
            generate also runlength distribution figures for each
            single function. Works only if more than two algorithms are given. 
            These figures are not (yet) used in the LaTeX templates. 
            
        --input-path=INPUTPATH
        
            all folder/file arguments are prepended with the given value
            which must be a valid path. 
            
        --in-a-hurry
        
            takes values between 0 (default) and 1000, fast processing that 
            does not write eps files and uses a small number of bootstrap samples

    Exceptions raised:
    
    *Usage* -- Gives back a usage message.

    Examples:

    * Calling the rungeneric.py interface from the command line::

        $ python bbob_pproc/rungeneric.py -v AMALGAM BIPOP-CMA-ES

    * Loading this package and calling the main from the command line
      (requires that the path to this package is in python search path)::

        $ python -m bbob_pproc.rungeneric -h

      This will print out this help message.

    * From the python interpreter (requires that the path to this
      package is in python search path)::

        >> import bbob_pproc as bb
        >> bb.rungeneric.main('-o outputfolder folder1 folder2'.split())

      This will execute the post-processing on the data found in
      :file:`folder1` and :file:`folder2`. The ``-o`` option changes the
      output folder from the default :file:`ppdata` to
      :file:`outputfolder`.

    """

    if argv is None:
        argv = sys.argv[1:]

    try:
        try:
            opts, args = getopt.getopt(argv, shortoptlist, longoptlist + ['omit-single', 'in-a-hurry=', 'input-path='])
        except getopt.error as msg:
            raise Usage(msg)

        if not (args):
            usage()
            sys.exit()

        verbose = False
        outputdir = 'ppdata'
        inputdir = '.'

        #Process options
        shortoptlist1 = list("-" + i.rstrip(":")
                             for i in _splitshortoptlist(rungeneric1.shortoptlist))
        shortoptlist2 = list("-" + i.rstrip(":")
                             for i in _splitshortoptlist(rungeneric2.shortoptlist))
        shortoptlistmany = list("-" + i.rstrip(":")
                                for i in _splitshortoptlist(rungenericmany.shortoptlist))
        shortoptlist1.remove("-o")
        shortoptlist2.remove("-o")
        shortoptlistmany.remove("-o")
        longoptlist1 = list( "--" + i.rstrip("=") for i in rungeneric1.longoptlist)
        longoptlist2 = list( "--" + i.rstrip("=") for i in rungeneric2.longoptlist)
        longoptlistmany = list( "--" + i.rstrip("=") for i in rungenericmany.longoptlist)

        genopts1 = []
        genopts2 = []
        genoptsmany = []
        for o, a in opts:
            if o in ("-h", "--help"):
                usage()
                sys.exit()
            elif o in ("-o", "--output-dir"):
                outputdir = a
            elif o in ("--in-a-hurry", ):
                genericsettings.in_a_hurry = int(a)
            elif o in ("--input-path", ):
                inputdir = a
            else:
                isAssigned = False
                if o in longoptlist1 or o in shortoptlist1:
                    genopts1.append(o)
                    # Append o and then a separately otherwise the list of
                    # command line arguments might be incorrect
                    if a:
                        genopts1.append(a)
                    isAssigned = True
                if o in longoptlist2 or o in shortoptlist2:
                    genopts2.append(o)
                    if a:
                        genopts2.append(a)
                    isAssigned = True
                if o in longoptlistmany or o in shortoptlistmany:
                    genoptsmany.append(o)
                    if a:
                        genoptsmany.append(a)
                    isAssigned = True
                if o in ("-v","--verbose"):
                    verbose = True
                    isAssigned = True
                if o in ('--omit-single', '--in-a-hurry'):
                    isAssigned = True
                if not isAssigned:
                    assert False, "unhandled option"

        if (not verbose):
            warnings.filterwarnings('module', '.*', UserWarning, '.*')
            warnings.simplefilter('ignore')  # that is bad, but otherwise to many warnings appear 

        print(("Post-processing: will generate output " +
               "data in folder %s" % outputdir))
        print("  this might take several minutes.")

        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
            if verbose:
                print('Folder %s was created.' % (outputdir))
        
        truncate_latex_command_file(os.path.join(outputdir, 'bbob_pproc_commands.tex'))

        for i in range(len(args)):  # prepend common path inputdir to all names
            args[i] = os.path.join(inputdir, args[i])

        for i, alg in enumerate(args):
            # remove '../' from algorithm output folder
            if len(args) == 1 or '--omit-single' not in dict(opts):
                tmpoutputdir = os.path.join(outputdir, alg.replace('..' + os.sep, '').lstrip(os.sep))
                rungeneric1.main(genopts1
                                + ["-o", tmpoutputdir, alg])
                prepend_to_file(os.path.join(outputdir, 'bbob_pproc_commands.tex'), 
                                ['\\providecommand{\\algfolder}{' + alg.replace('..' + os.sep, '').rstrip(os.sep).replace(os.sep, '/') + '/}'])

        if len(args) == 2:
            rungeneric2.main(genopts2 + ["-o", outputdir] + args)
        elif len(args) > 2:
            rungenericmany.main(genoptsmany + ["-o", outputdir] + args)

        open(os.path.join(outputdir, 'bbob_pproc_commands.tex'), 'a').close() 
        
        print_done()

    #TODO prevent loading the data every time...

    except Usage as err:
        print(err.msg, file=sys.stderr)
        print("for help use -h or --help", file=sys.stderr)
        return 2

if __name__ == "__main__":
    res = main()
    if genericsettings.test: 
        print(res)
    sys.exit(res)
