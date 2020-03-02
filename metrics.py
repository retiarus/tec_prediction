from numpy import sqrt

def rms(data, axis=None):
    """Compute RMS."""
    if (axis is None):
        return sqrt((data**2).mean())
    else:
        return sqrt((data**2).mean(axis=axis))
