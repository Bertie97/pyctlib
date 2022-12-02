import numpy as np
import torch
from torch import Tensor
from typing import Tuple, Union, Dict, Any, List
import builtins
from operator import index
from collections import namedtuple

def _bin_edges(sample, bins=None, range=None):
    """ Create edge arrays
    """
    Dlen, Ndim = sample.shape

    nbin = np.empty(Ndim, int)    # Number of bins in each dimension
    edges = Ndim * [None]         # Bin edges for each dim (will be 2D array)
    dedges = Ndim * [None]        # Spacing between edges (will be 2D array)

    # Select range for each dimension
    # Used only if number of bins is given.
    if range is None:
        smin = np.atleast_1d(np.array(sample.min(axis=0), float))
        smax = np.atleast_1d(np.array(sample.max(axis=0), float))
    else:
        if len(range) != Ndim:
            raise ValueError(
                f"range given for {len(range)} dimensions; {Ndim} required")
        smin = np.empty(Ndim)
        smax = np.empty(Ndim)
        for i in builtins.range(Ndim):
            if range[i][1] < range[i][0]:
                raise ValueError(
                    "In {}range, start must be <= stop".format(
                        f"dimension {i + 1} of " if Ndim > 1 else ""))
            smin[i], smax[i] = range[i]

    # Make sure the bins have a finite width.
    for i in builtins.range(len(smin)):
        if smin[i] == smax[i]:
            smin[i] = smin[i] - .5
            smax[i] = smax[i] + .5

    # Preserve sample floating point precision in bin edges
    edges_dtype = (sample.dtype if np.issubdtype(sample.dtype, np.floating)
                   else float)

    # Create edge arrays
    for i in builtins.range(Ndim):
        if np.isscalar(bins[i]):
            nbin[i] = bins[i] + 2  # +2 for outlier bins
            edges[i] = np.linspace(smin[i], smax[i], nbin[i] - 1,
                                   dtype=edges_dtype)
        else:
            edges[i] = np.asarray(bins[i], edges_dtype)
            nbin[i] = len(edges[i]) + 1  # +1 for outlier bins
        dedges[i] = np.diff(edges[i])

    nbin = np.asarray(nbin)

    return nbin, edges, dedges

def _bin_numbers(sample, nbin, edges, dedges):
    """Compute the bin number each sample falls into, in each dimension
    """
    Dlen, Ndim = sample.shape

    sampBin = [
        np.digitize(sample[:, i], edges[i])
        for i in builtins.range(Ndim)
    ]

    # Using `digitize`, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right
    # edge to be counted in the last bin, and not as an outlier.
    for i in builtins.range(Ndim):
        # Find the rounding precision
        dedges_min = dedges[i].min()
        if dedges_min == 0:
            raise ValueError('The smallest edge difference is numerically 0.')
        decimal = int(-np.log10(dedges_min)) + 6
        # Find which points are on the rightmost edge.
        on_edge = np.where(np.around(sample[:, i], decimal) ==
                           np.around(edges[i][-1], decimal))[0]
        # Shift these points one bin to the left.
        sampBin[i][on_edge] -= 1

    # Compute the sample indices in the flattened statistic matrix.
    binnumbers = np.ravel_multi_index(sampBin, nbin)

    return binnumbers

BinnedStatisticddResult = namedtuple('BinnedStatisticddResult',
                                     ('statistic', 'bin_edges',
                                      'binnumber'))

def binned_statistic_dd(sample, values, statistic='mean',
                        bins=10, range=None, expand_binnumbers=False,
                        binned_statistic_result=None):
    """
    Compute a multidimensional binned statistic for a set of data.
    This is a generalization of a histogramdd function.  A histogram divides
    the space into bins, and returns the count of the number of points in
    each bin.  This function allows the computation of the sum, mean, median,
    or other statistic of the values within each bin.
    Parameters
    ----------
    sample : array_like
        Data to histogram passed as a sequence of N arrays of length D, or
        as an (N,D) array.
    values : (N,) array_like or list of (N,) array_like
        The data on which the statistic will be computed.  This must be
        the same shape as `sample`, or a list of sequences - each with the
        same shape as `sample`.  If `values` is such a list, the statistic
        will be computed on each independently.
    statistic : string or callable, optional
        The statistic to compute (default is 'mean').
        The following statistics are available:
          * 'mean' : compute the mean of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'median' : compute the median of values for points within each
            bin. Empty bins will be represented by NaN.
          * 'count' : compute the count of points within each bin.  This is
            identical to an unweighted histogram.  `values` array is not
            referenced.
          * 'sum' : compute the sum of values for points within each bin.
            This is identical to a weighted histogram.
          * 'std' : compute the standard deviation within each bin. This
            is implicitly calculated with ddof=0. If the number of values
            within a given bin is 0 or 1, the computed standard deviation value
            will be 0 for the bin.
          * 'min' : compute the minimum of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'max' : compute the maximum of values for point within each bin.
            Empty bins will be represented by NaN.
          * function : a user-defined function which takes a 1D array of
            values, and outputs a single numerical statistic. This function
            will be called on the values in each bin.  Empty bins will be
            represented by function([]), or NaN if this returns an error.
    bins : sequence or positive int, optional
        The bin specification must be in one of the following forms:
          * A sequence of arrays describing the bin edges along each dimension.
          * The number of bins for each dimension (nx, ny, ... = bins).
          * The number of bins for all dimensions (nx = ny = ... = bins).
    range : sequence, optional
        A sequence of lower and upper bin edges to be used if the edges are
        not given explicitly in `bins`. Defaults to the minimum and maximum
        values along each dimension.
    expand_binnumbers : bool, optional
        'False' (default): the returned `binnumber` is a shape (N,) array of
        linearized bin indices.
        'True': the returned `binnumber` is 'unraveled' into a shape (D,N)
        ndarray, where each row gives the bin numbers in the corresponding
        dimension.
        See the `binnumber` returned value, and the `Examples` section of
        `binned_statistic_2d`.
    binned_statistic_result : binnedStatisticddResult
        Result of a previous call to the function in order to reuse bin edges
        and bin numbers with new values and/or a different statistic.
        To reuse bin numbers, `expand_binnumbers` must have been set to False
        (the default)
        .. versionadded:: 0.17.0
    Returns
    -------
    statistic : ndarray, shape(nx1, nx2, nx3,...)
        The values of the selected statistic in each two-dimensional bin.
    bin_edges : list of ndarrays
        A list of D arrays describing the (nxi + 1) bin edges for each
        dimension.
    binnumber : (N,) array of ints or (D,N) ndarray of ints
        This assigns to each element of `sample` an integer that represents the
        bin in which this observation falls.  The representation depends on the
        `expand_binnumbers` argument.  See `Notes` for details.
    See Also
    --------
    numpy.digitize, numpy.histogramdd, binned_statistic, binned_statistic_2d
    Notes
    -----
    Binedges:
    All but the last (righthand-most) bin is half-open in each dimension.  In
    other words, if `bins` is ``[1, 2, 3, 4]``, then the first bin is
    ``[1, 2)`` (including 1, but excluding 2) and the second ``[2, 3)``.  The
    last bin, however, is ``[3, 4]``, which *includes* 4.
    `binnumber`:
    This returned argument assigns to each element of `sample` an integer that
    represents the bin in which it belongs.  The representation depends on the
    `expand_binnumbers` argument. If 'False' (default): The returned
    `binnumber` is a shape (N,) array of linearized indices mapping each
    element of `sample` to its corresponding bin (using row-major ordering).
    If 'True': The returned `binnumber` is a shape (D,N) ndarray where
    each row indicates bin placements for each dimension respectively.  In each
    dimension, a binnumber of `i` means the corresponding value is between
    (bin_edges[D][i-1], bin_edges[D][i]), for each dimension 'D'.
    .. versionadded:: 0.11.0
    Examples
    --------
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    Take an array of 600 (x, y) coordinates as an example.
    `binned_statistic_dd` can handle arrays of higher dimension `D`. But a plot
    of dimension `D+1` is required.
    >>> mu = np.array([0., 1.])
    >>> sigma = np.array([[1., -0.5],[-0.5, 1.5]])
    >>> multinormal = stats.multivariate_normal(mu, sigma)
    >>> data = multinormal.rvs(size=600, random_state=235412)
    >>> data.shape
    (600, 2)
    Create bins and count how many arrays fall in each bin:
    >>> N = 60
    >>> x = np.linspace(-3, 3, N)
    >>> y = np.linspace(-3, 4, N)
    >>> ret = stats.binned_statistic_dd(data, np.arange(600), bins=[x, y],
    ...                                 statistic='count')
    >>> bincounts = ret.statistic
    Set the volume and the location of bars:
    >>> dx = x[1] - x[0]
    >>> dy = y[1] - y[0]
    >>> x, y = np.meshgrid(x[:-1]+dx/2, y[:-1]+dy/2)
    >>> z = 0
    >>> bincounts = bincounts.ravel()
    >>> x = x.ravel()
    >>> y = y.ravel()
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> with np.errstate(divide='ignore'):   # silence random axes3d warning
    ...     ax.bar3d(x, y, z, dx, dy, bincounts)
    Reuse bin numbers and bin edges with new values:
    >>> ret2 = stats.binned_statistic_dd(data, -np.arange(600),
    ...                                  binned_statistic_result=ret,
    ...                                  statistic='mean')
    """
    known_stats = ['mean', 'median', 'count', 'sum', 'std', 'min', 'max']
    if not callable(statistic) and statistic not in known_stats:
        raise ValueError('invalid statistic %r' % (statistic,))

    try:
        bins = index(bins)
    except TypeError:
        # bins is not an integer
        pass
    # If bins was an integer-like object, now it is an actual Python int.

    # NOTE: for _bin_edges(), see e.g. gh-11365
    if isinstance(bins, int) and not np.isfinite(sample).all():
        raise ValueError('%r contains non-finite values.' % (sample,))

    # `Ndim` is the number of dimensions (e.g. `2` for `binned_statistic_2d`)
    # `Dlen` is the length of elements along each dimension.
    # This code is based on np.histogramdd
    try:
        # `sample` is an ND-array.
        Dlen, Ndim = sample.shape
    except (AttributeError, ValueError):
        # `sample` is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        Dlen, Ndim = sample.shape

    # Store initial shape of `values` to preserve it in the output
    values = np.asarray(values)
    input_shape = list(values.shape)
    # Make sure that `values` is 2D to iterate over rows
    values = np.atleast_2d(values)
    Vdim, Vlen = values.shape

    # Make sure `values` match `sample`
    if(statistic != 'count' and Vlen != Dlen):
        raise AttributeError('The number of `values` elements must match the '
                             'length of each `sample` dimension.')

    try:
        M = len(bins)
        if M != Ndim:
            raise AttributeError('The dimension of bins must be equal '
                                 'to the dimension of the sample x.')
    except TypeError:
        bins = Ndim * [bins]

    if binned_statistic_result is None:
        nbin, edges, dedges = _bin_edges(sample, bins, range)
        binnumbers = _bin_numbers(sample, nbin, edges, dedges)
    else:
        edges = binned_statistic_result.bin_edges
        nbin = np.array([len(edges[i]) + 1 for i in builtins.range(Ndim)])
        # +1 for outlier bins
        dedges = [np.diff(edges[i]) for i in builtins.range(Ndim)]
        binnumbers = binned_statistic_result.binnumber

    result = np.empty([Vdim, nbin.prod()], float)

    if statistic == 'mean':
        result.fill(np.nan)
        flatcount = np.bincount(binnumbers, None)
        a = flatcount.nonzero()
        for vv in builtins.range(Vdim):
            flatsum = np.bincount(binnumbers, values[vv])
            result[vv, a] = flatsum[a] / flatcount[a]
    elif statistic == 'std':
        result.fill(0)
        _calc_binned_statistic(Vdim, binnumbers, result, values, np.std)
    elif statistic == 'count':
        result.fill(0)
        flatcount = np.bincount(binnumbers, None)
        a = np.arange(len(flatcount))
        result[:, a] = flatcount[np.newaxis, :]
    elif statistic == 'sum':
        result.fill(0)
        for vv in builtins.range(Vdim):
            flatsum = np.bincount(binnumbers, values[vv])
            a = np.arange(len(flatsum))
            result[vv, a] = flatsum
    elif statistic == 'median':
        result.fill(np.nan)
        _calc_binned_statistic(Vdim, binnumbers, result, values, np.median)
    elif statistic == 'min':
        result.fill(np.nan)
        _calc_binned_statistic(Vdim, binnumbers, result, values, np.min)
    elif statistic == 'max':
        result.fill(np.nan)
        _calc_binned_statistic(Vdim, binnumbers, result, values, np.max)
    elif callable(statistic):
        with np.errstate(invalid='ignore'), suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            try:
                null = statistic([])
            except Exception:
                null = np.nan
        result.fill(null)
        _calc_binned_statistic(Vdim, binnumbers, result, values, statistic,
                               is_callable=True)

    # Shape into a proper matrix
    result = result.reshape(np.append(Vdim, nbin))

    # Remove outliers (indices 0 and -1 for each bin-dimension).
    core = tuple([slice(None)] + Ndim * [slice(1, -1)])
    result = result[core]

    # Unravel binnumbers into an ndarray, each row the bins for each dimension
    if(expand_binnumbers and Ndim > 1):
        binnumbers = np.asarray(np.unravel_index(binnumbers, nbin))

    if np.any(result.shape[1:] != nbin - 2):
        raise RuntimeError('Internal Shape Error')

    # Reshape to have output (`result`) match input (`values`) shape
    result = result.reshape(input_shape[:-1] + list(nbin-2))

    return BinnedStatisticddResult(result, edges, binnumbers)

class bined_statistic_2d:

    def __init__(self, statistic="mean", bins=10, drange=None):
        self.statistic = statistic
        assert statistic in ["mean", "max", "sum", "count", "min"]
        if isinstance(bins, int):
            self.xbins = self.ybins = bins
        elif isinstance(bins, list) or isinstance(bins, tuple):
            assert len(bins) == 2
            self.xbins, self.ybins == bins
        else:
            raise TypeError
        self.histogram = None
        assert drange is not None
        self.drange = drange
        self.N = 0

    def update(self, xs, ys, values):
        assert len(xs) == len(ys)
        n = len(xs)
        if isinstance(values, np.ndarray) and values.ndim == 2:
            values = [values[:, index] for index in range(values.shape[1])]
        if self.statistic == "mean":
            sum = binned_statistic_dd([xs, ys], values, statistic="sum", range=self.drange, bins=[self.xbins, self.ybins])[0]
            count = binned_statistic_dd([xs, ys], values, statistic="count", range=self.drange, bins=[self.xbins, self.ybins])[0]
        else:
            result = binned_statistic_dd([xs, ys], values, statistic=self.statistic, range=self.drange, bins=[self.xbins, self.ybins])[0]
        if self.histogram is None:
            if self.statistic == "mean":
                self.history_sum = sum
                self.history_count = count
                with np.errstate(divide='ignore', invalid='ignore'):
                    self.histogram = np.divide(self.history_sum, self.history_count)
                return self.histogram
            else:
                self.histogram = result
                return self.histogram
        elif self.statistic == "mean":
            new_sum = sum + self.history_sum
            new_count = count + self.history_count
            self.history_sum = new_sum
            self.history_count = new_count
            with np.errstate(divide='ignore', invalid='ignore'):
                self.histogram = np.divide(self.history_sum, self.history_count)
        elif self.statistic == "max":
            self.histogram = np.nanmax(np.array([result, self.histogram]), axis=0)
        elif self.statistic == "min":
            self.histogram = np.nanmax(np.array([result, self.histogram]), axis=0)
        elif self.statistic == "count":
            self.histogram = result + self.histogram
        elif self.statistic == "sum":
            self.histogram = result + self.histogram
        return self.histogram

    def clear(self):
        self.histogram = None
