#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generic routines for figure generation."""
from operator import itemgetter
from itertools import groupby
import warnings
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace
from bbob_pproc import genericsettings, toolsstats

xrange = range

def saveFigure(filename, figFormat=genericsettings.fig_formats, verbose=True):
    """Save figure into an image file."""

    if isinstance(figFormat, str):
        try:
            plt.savefig(filename + '.' + figFormat, dpi = 60 if genericsettings.in_a_hurry else 300,
                        format=figFormat)
            if verbose:
                print(('Wrote figure in %s.' % (filename + '.' + figFormat)))
        except IOError:
            warnings.warn('%s is not writeable.' % (filename + '.' + figFormat))
    else:
        #if not isinstance(figFormat, basestring):
        for entry in figFormat:
            try:
                plt.savefig(filename + '.' + entry, dpi = 60 if genericsettings.in_a_hurry else 300,
                            format=entry)
                if verbose:
                    print(('Wrote figure in %s.' %(filename + '.' + entry)))
            except IOError:
                warnings.warn('%s is not writeable.' % (filename + '.' + entry))

def plotUnifLogXMarkers(x, y, nbperdecade, logscale=False, **kwargs):
    """Proxy plot function: markers are evenly spaced on the log x-scale

    This method generates plots with markers regularly spaced on the
    x-scale whereas the matplotlib.pyplot.plot function will put markers
    on data points.

    This method outputs a list of three lines.Line2D objects: the first
    with the line style, the second for the markers and the last for the
    label.

    This function only works with monotonous graph.

    """
    res = plt.plot(x, y, **kwargs)

    def marker_positions(xdata, ydata, nbperdecade, maxnb, ax):
        """replacement for downsample with at most 12 points"""
        if 11 < 3:
            return old_downsample(xdata, ydata)
        tfy = np.log10 if logscale else lambda x: x
            
        xdatarange = np.log10(max([max(xdata), ax[0], ax[1]]) + 0.5) - np.log10(min([min(xdata), ax[0], ax[1]]) + 0.5)  #np.log10(xdata[-1]) - np.log10(xdata[0])
        ydatarange = tfy(max([max(ydata), ax[2], ax[3]]) + 0.5) - tfy(min([min(ydata), ax[2], ax[3]]) + 0.5)  # tfy(ydata[-1]) - tfy(ydata[0])
        nbmarkers = np.min([maxnb, nbperdecade + np.ceil(nbperdecade * (1e-99 + np.abs(np.log10(max(xdata)) - np.log10(min(xdata)))))])
        probs = np.abs(np.diff(np.log10(xdata)))/xdatarange + np.abs(np.diff(tfy(ydata)))/ydatarange
        xpos = []
        ypos= []
        if sum(probs) > 0:
            xoff = np.random.rand() / nbmarkers
            probs /= sum(probs)
            cum = np.cumsum(probs)
            for xact in np.arange(0, 1, 1./nbmarkers):
                pos = xoff + xact + (1./nbmarkers) * (0.3 + 0.4 * np.random.rand())
                idx = np.abs(cum - pos).argmin()  # index of closest value
                xpos.append(xdata[idx])
                ypos.append(ydata[idx])
        xpos.append(xdata[-1])
        ypos.append(ydata[-1])
        return xpos, ypos
    
    def old_downsample(xdata, ydata):
        """Downsample arrays of data, superseeded by method marker_position
        
        From xdata and ydata return x and y which have only nbperdecade
        elements times the number of decades in xdata.

        """
        # powers of ten 10**(i/nbperdecade)
        # get segments coordinates x1, x2, y1, y2
        # Add data at the front and the back,
        # otherwise the line of markers is prolonged at the y-position
        # of the first and last marker which may not correspond to the
        # 1st and last y-value
        if 'steps' in plt.getp(res[0], 'drawstyle'): # other conditions?
            #xdata = np.hstack((10 ** (np.floor(np.log10(xdata[0]) * nbperdecade) / nbperdecade),
            #                   xdata,
            #                   10 ** (np.ceil(np.log10(xdata[-1]) * nbperdecade) / nbperdecade)))
            #ydata = np.hstack((ydata[0], ydata, ydata[-1]))
            # Add data only at the back
            xdata = np.hstack((xdata,
                               10 ** (np.ceil(np.log10(xdata[-1]) * nbperdecade) / nbperdecade)))
            ydata = np.hstack((ydata, ydata[-1]))

        tmpdata = np.column_stack((xdata, ydata))
        it = groupby(tmpdata, lambda x: x[0])
        seg = []
        try:
            k0, g0 = next(it)
            g0 = np.vstack(g0)[:, 1]
            while True:
                if len(g0) > 1:
                    seg.append(((k0, k0), (min(g0), max(g0))))
                k, g = next(it)
                g = np.vstack(g)[:, 1]
                seg.append(((k0, k), (g0[-1], g[0])))
                k0 = k
                g0 = g
        except StopIteration:
            pass
        downx = []
        downy = []
        for segx, segy in seg:
            minidx = np.ceil(np.log10(min(segx[0], segx[1])) * nbperdecade)
            maxidx = np.floor(np.log10(max(segx[0], segx[1])) * nbperdecade)
            intermx = 10. ** (np.arange(minidx, maxidx + 1) / nbperdecade)
            downx.extend(intermx)
            if plt.getp(res[0], 'drawstyle') in ('steps', 'steps-pre'):
                downy.extend(len(intermx) * [max(segy[0], segy[1])])
            elif plt.getp(res[0], 'drawstyle') == 'steps-post':
                downy.extend(len(intermx) * [min(segy[0], segy[1])])
            elif plt.getp(res[0], 'drawstyle') == 'steps-mid':
                if logscale:
                    ymid = 10. ** ((np.log10(segy[0]) + np.log10(segy[1])) / 2.)
                else:
                    ymid = (segy[0] + segy[1]) / 2.
                downy.extend(len(intermx) * [ymid])
            elif plt.getp(res[0], 'drawstyle') == 'default':
                # log interpolation / semi-log
                dlgx = np.log10(segx[1]) - np.log10(segx[0])
                if logscale:
                    tmp = 10.**(np.log10(segy[0]) + (np.log10(intermx) - np.log10(segx[0])) * (np.log10(segy[1]) - np.log10(segy[0])) / dlgx)
                else:
                    tmp = segy[0] + (np.log10(intermx) - np.log10(segx[0])) * (segy[1] - segy[0]) / dlgx
                downy.extend(tmp)
        resdownx = []
        resdowny = []
        tmpdata = np.column_stack((downx, downy))
        it = groupby(tmpdata, lambda x: x[0])
        try:
            while True:
                k, g = next(it)
                g = np.vstack(g)[:, 1]
                resdownx.append(k)
                resdowny.append(10.**((np.log10(min(g)) + np.log10(max(g))) / 2.))
        except StopIteration:
            pass
        return resdownx, resdowny

    if 'marker' in kwargs and len(x) > 0:
        # x2, y2 = downsample(x, y)
        x2, y2 = marker_positions(x, y, nbperdecade, 19, plt.axis())
        try:
            res2 = plt.plot(x2, y2)
        except ValueError:
            raise # TODO
        for i in res2:
            i.update_from(res[0]) # copy all attributes of res
        plt.setp(res2, linestyle='', label='')
        res.extend(res2)

    if 'label' in kwargs:
        res3 = plt.plot([], [], **kwargs)
        for i in res3:
            i.update_from(res[0]) # copy all attributes of res
        res.extend(res3)

    plt.setp(res[0], marker='', label='')
    return res

def consecutiveNumbers(data):
    """Groups a sequence of integers into ranges of consecutive numbers.

    Example::
      >>> import sys
      >>> import os
      >>> os.chdir(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
      >>> import bbob_pproc as bb
      >>> bb.ppfig.consecutiveNumbers([0, 1, 2, 4, 5, 7, 8, 9])
      '0-2, 4, 5, 7-9'

    Range of consecutive numbers is at least 3 (therefore [4, 5] is
    represented as "4, 5").

    """
    res = []
    tmp = groupByRange(data)
    for i in tmp:
        tmpstring = list(str(j) for j in i)
        if len(i) <= 2 : # This means length of ranges are at least 3
            res.append(', '.join(tmpstring))
        else:
            res.append('-'.join((tmpstring[0], tmpstring[-1])))

    return ', '.join(res)

def groupByRange(data):
    """Groups a sequence of integers into ranges of consecutive numbers.

    Helper function of consecutiveNumbers(data), returns a list of lists.
    The key to the solution is differencing with a range so that
    consecutive numbers all appear in same group.
    Useful for determining ranges of functions.
    Ref: http://docs.python.org/release/3.0.1/library/itertools.html

    """
    res = []
    for _k, g in groupby(enumerate(data), lambda i_x:i_x[0]-i_x[1]):
        res.append(list(i for i in map(itemgetter(1), g)))

    return res

def logxticks(limits=[-np.inf, np.inf]):
    """Modify log-scale figure xticks from 10^i to i for values with the ``limits``.
    
    This is to have xticks that are more visible.
    Modifying the x-limits of the figure after calling this method will
    not update the ticks.
    Please make sure the xlabel is changed accordingly.
    
    """
    _xticks = plt.xticks()
    _xticks
    newxticks = []
    for j in _xticks[0]:
        if j > limits[0] and j < limits[1]: # tick annotations only within the limits
            newxticks.append('%d' % round(np.log10(j)))
        else:
            newxticks.append('')
    plt.xticks(_xticks[0], newxticks)
    # TODO: check the xlabel is changed accordingly?

def beautify():
    """ Customize a figure by adding a legend, axis label, etc."""
    # TODO: what is this function for?
    # Input checking

    # Get axis handle and set scale for each axis
    axisHandle = plt.gca()
    axisHandle.set_yscale("log")

    # Grid options
    axisHandle.grid(True)

    _ymin, ymax = plt.ylim()
    plt.ylim(ymin=10**-0.2, ymax=ymax) # Set back the default maximum.

    tmp = axisHandle.get_yticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(np.log10(i)))
    axisHandle.set_yticklabels(tmp2)
    axisHandle.set_ylabel('log10 of ERT')

def generateData(dataSet, targetFuncValue):
    """Returns an array of results to be plotted.

    1st column is ert, 2nd is  the number of success, 3rd the success
    rate, 4th the sum of the number of  function evaluations, and
    finally the median on successful runs.

    """
    res = []
    data = []

    it = iter(reversed(dataSet.evals))
    i = next(it)
    prev = np.array([np.nan] * len(i))

    while i[0] <= targetFuncValue:
        prev = i
        try:
            i = next(it)
        except StopIteration:
            break

    data = prev[1:].copy() # keep only the number of function evaluations.
    succ = (np.isnan(data) == False)
    if succ.any():
        med = toolsstats.prctile(data[succ], 50)[0]
        #Line above was modified at rev 3050 to make sure that we consider only
        #successful trials in the median
    else:
        med = np.nan

    data[np.isnan(data)] = dataSet.maxevals[np.isnan(data)]

    res = []
    res.extend(toolsstats.sp(data, issuccessful=succ, allowinf=False))
    res.append(np.mean(data)) #mean(FE)
    res.append(med)

    return np.array(res)

def plot(dsList, _valuesOfInterest=(10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8),
         isbyinstance=True, kwargs={}):
    """From a DataSetList, plot a graph. Not in use and superseeded by ppfigdim.main!?"""

    #set_trace()
    res = []

    valuesOfInterest = list(_valuesOfInterest)
    valuesOfInterest.sort(reverse=True)

    def transform(dsList):
        """Create dictionary of instances."""

        class StrippedUpDS():
            """Data Set stripped up of everything."""

            pass

        res = {}
        for i in dsList:
            dictinstance = i.createDictInstance()
            for j, idx in list(dictinstance.items()):
                tmp = StrippedUpDS()
                idxs = list(k + 1 for k in idx)
                idxs.insert(0, 0)
                tmp.evals = i.evals[:, np.r_[idxs]].copy()
                tmp.maxevals = i.maxevals[np.ix_(idx)].copy()
                res.setdefault(j, [])
                res.get(j).append(tmp)
        return res
    
    for i in range(len(valuesOfInterest)):

        succ = []
        unsucc = []
        displaynumber = []
        data = []

        dictX = transform(dsList)
        for x in sorted(dictX.keys()):
            dsListByX = dictX[x]
            for j in dsListByX:
                tmp = generateData(j, valuesOfInterest[i])
                if tmp[2] > 0: #Number of success is larger than 0
                    succ.append(np.append(x, tmp))
                    if tmp[2] < j.nbRuns():
                        displaynumber.append((x, tmp[0], tmp[2]))
                else:
                    unsucc.append(np.append(x, tmp))

        if succ:
            tmp = np.vstack(succ)
            #ERT
            res.extend(plt.plot(tmp[:, 0], tmp[:, 1], **kwargs))
            #median
            tmp2 = plt.plot(tmp[:, 0], tmp[:, -1], **kwargs)
            plt.setp(tmp2, linestyle='', marker='+', markersize=30, markeredgewidth=5)
            #, color=colors[i], linestyle='', marker='+', markersize=30, markeredgewidth=5))
            res.extend(tmp2)

        # To have the legend displayed whatever happens with the data.
        tmp = plt.plot([], [], **kwargs)
        plt.setp(tmp, label=' %+d' % (np.log10(valuesOfInterest[i])))
        res.extend(tmp)

        #Only for the last target function value
        if unsucc:
            tmp = np.vstack(unsucc) # tmp[:, 0] needs to be sorted!
            res.extend(plt.plot(tmp[:, 0], tmp[:, 1], **kwargs))

    if displaynumber: # displayed only for the smallest valuesOfInterest
        for j in displaynumber:
            t = plt.text(j[0], j[1]*1.85, "%.0f" % j[2],
                         horizontalalignment="center",
                         verticalalignment="bottom")
            res.append(t)

    return res
