import numpy as np
import scipy.ndimage
from . import _perlin

XYSCALEBASE = 100

def gausswin_matlab(L: int, alpha: float) -> np.ndarray:
    """MATLAB-compatible Gaussian window"""
    n = np.arange(L, dtype=np.float32)
    N = float(L - 1) / 2.0 if L > 1 else 1.0
    x = (n - N) / (N if N != 0 else 1.0)
    return np.exp(-0.5 * (alpha * x) ** 2).astype(np.float32)

def upsample_1d_along_axis(x: np.ndarray, factor: int, phase: int = 0, axis: int = 0) -> np.ndarray:
    """MATLAB upsample: result length = len(x) * factor; nonzeros at indices phase, phase+factor, ..."""
    x = np.asarray(x)
    assert factor >= 1 and 0 <= phase < factor
    new_shape = list(x.shape)
    new_shape[axis] = x.shape[axis] * factor
    y = np.zeros(new_shape, dtype=x.dtype)
    sl = [slice(None)] * y.ndim
    sl[axis] = slice(phase, None, factor)
    y[tuple(sl)] = x
    return y

def frozen_upscale(img: np.ndarray, factor: int) -> np.ndarray:
    """Trippy-style frozen Gaussian upsampler for ultra-smooth interpolation"""
    out = img.astype(np.float32, copy=True)
    for _ in range(2):
        out = out.T
        ph = int(np.round(factor / 2.0))
        out = upsample_1d_along_axis(out, factor=factor, phase=ph, axis=0)
        L = out.shape[0]
        alpha = (np.sqrt(0.5) * L) / float(factor)
        k = gausswin_matlab(L, alpha)
        k = np.fft.ifftshift((factor / np.sum(k)) * k.astype(np.float32))
        Kf = np.fft.fft(k, n=L).astype(np.complex64)
        Xf = np.fft.fft(out, axis=0)
        Yf = Kf[:, None] * Xf
        out = np.fft.ifft(Yf, axis=0).real.astype(np.float32)
    return out

def filter_frames(im, filt, *args):
    """Apply a filter/transformation to an image batch

    Parameters
    ----------
    im : 3D float ndarray, values ∈ [0,1]
        Frames to filter
    filt : str
        The name of the filter
    *args : tuple
        Extra arguments are passed to the filter

    Returns
    -------
    im : 3D float ndarray, values ∈ [0,1]
        Filtered noise movie
    """
    if filt == "threshold":
        return (im>args[0]).astype(np.float32)
    if filt == "softthresh":
        return 1/(1+np.exp(-args[0]*(im-.5)))
    if filt == "comb":
        return (im//args[0] % 2 == 1).astype(np.float32)
    if filt == "invert":
        return 1-im
    if filt == "reverse":
        return im # We need to use filter_index_function for this
    if filt == "blur":
        return np.asarray([scipy.ndimage.filters.gaussian_filter(im.astype(np.float32)[:,:,i], args[0], mode='wrap') for i in range(0, im.shape[2])]).transpose([1,2,0])
    if filt == "wood":
        return (im % args[0]) / args[0]
    if filt == "center":
        return 1-(np.abs(im-.5)*2)
    if filt == "photodiode":
        im = im.copy()
        s = args[0]
        im[:s,-s:,::2] = 0
        im[:s,-s:,1::2] = 1
        return im
    if filt == "photodiode_anywhere":
        im = im.copy()
        x = args[0]
        y = args[1]
        s = args[2]
        im[y:(y+s),x:(x+s),::2] = 0
        im[y:(y+s),x:(x+s),1::2] = 1
        return im
    if filt == "photodiode_b2":
        im = im.copy()
        s = 125
        im[:s,-s:,::2] = 0
        im[:s,-s:,1::2] = 1
        return im
    if filt == "photodiode_fusi":
        im = im.copy()
        s = 75
        im[:s,-s:,::2] = 0
        im[:s,-s:,1::2] = 1
        return im
    if filt == "photodiode_bscope":
        im = im.copy()
        s = 100
        im[-s:,:s,::2] = 0
        im[-s:,:s,1::2] = 1
        return im
    if filt == "trippy_smooth":
        # Apply trippy-style smoothing with optional downsampling factor
        factor = args[0] if args else 4
        h, w, t = im.shape
        # Downsample first, then upsample with frozen Gaussian
        down_h, down_w = h // factor, w // factor
        smoothed = np.zeros_like(im)
        for i in range(t):
            # Simple downsampling by averaging blocks
            frame = im[:,:,i]
            down_frame = np.zeros((down_h, down_w), dtype=np.float32)
            for y in range(down_h):
                for x in range(down_w):
                    down_frame[y,x] = np.mean(frame[y*factor:(y+1)*factor, x*factor:(x+1)*factor])
            # Upsample with frozen Gaussian
            up_frame = frozen_upscale(down_frame, factor)
            smoothed[:,:,i] = up_frame[:h, :w]
        return smoothed
    if filt == "temporal_smooth":
        # Apply Hanning window smoothing across time (trippy-style)
        kernel_len = args[0] if args else 11
        if kernel_len % 2 == 0:
            kernel_len += 1  # Make odd
        kernel = np.hanning(kernel_len).astype(np.float32)
        kernel = kernel / np.sum(kernel)
        
        # Pad temporally and convolve
        pad_width = kernel_len // 2
        padded = np.pad(im, ((0,0), (0,0), (pad_width, pad_width)), mode='edge')
        smoothed = np.zeros_like(im)
        
        for i in range(im.shape[2]):
            for j in range(kernel_len):
                smoothed[:,:,i] += kernel[j] * padded[:,:,i+j]
        return smoothed
    if filt == "trippy_zebra":
        # Trippy-style smooth zebra: smooth first, then soft threshold
        smooth_factor = args[0] if len(args) > 0 else 4
        comb_freq = args[1] if len(args) > 1 else 0.08
        sigmoid_temp = args[2] if len(args) > 2 else 10.0
        
        # First apply spatial smoothing
        h, w, t = im.shape
        down_h, down_w = max(1, h // smooth_factor), max(1, w // smooth_factor)
        smoothed = np.zeros_like(im)
        
        for i in range(t):
            frame = im[:,:,i]
            # Downsample by averaging
            down_frame = np.zeros((down_h, down_w), dtype=np.float32)
            for y in range(down_h):
                for x in range(down_w):
                    y_start, y_end = y * smooth_factor, min((y+1) * smooth_factor, h)
                    x_start, x_end = x * smooth_factor, min((x+1) * smooth_factor, w)
                    down_frame[y,x] = np.mean(frame[y_start:y_end, x_start:x_end])
            
            # Upsample with frozen Gaussian
            if smooth_factor > 1:
                up_frame = frozen_upscale(down_frame, smooth_factor)
                smoothed[:,:,i] = up_frame[:h, :w]
            else:
                smoothed[:,:,i] = down_frame
        
        # Apply comb pattern with sigmoid (soft threshold)
        comb_phase = smoothed / comb_freq
        comb_raw = (comb_phase % 2.0) - 1.0  # Range [-1, 1]
        return 1.0 / (1.0 + np.exp(-sigmoid_temp * comb_raw))
    if callable(filt):
        return filt(im)
    raise ValueError("Invalid filter specified")

def apply_filters(arr, filters):
    for f in filters:
        if isinstance(f, str):
            n = f
            args = []
        else:
            n = f[0]
            args = f[1:]
        arr = filter_frames(arr, n, *args)
    return arr


def filter_frames_index_function(filters, nframes):
    """Reordering frames in the video based on the filter.


    Parameters
    ----------
    filters : list of strings or tuples
        the list of filters passed to save_video

    Returns
    -------
    function mapping int -> int
        Reindexing function

    Notes
    -----
    Some filters may need to operate on the global video instead of in
    batches.  However, for large videos, batches are necessary due to
    limited amounts of RAM.  Thus, this function should return another
    function which takes an index as input and outputs a new index,
    remapping the initial noise frame to the output video frame.  This
    was primarily designed to support reversing the video, but it might be
    useful for other things too.

    """
    if "reverse" in filters:
        return lambda x : nframes - x - 1
    return lambda x : x

def discretize(im):
    """Convert movie to an unsigned 8-bit integer

    Parameters
    ----------
    im : 3D float ndarray, values ∈ [0,1]
        Noise movie

    Returns
    -------
    3D int ndarray, values ∈ [0,255]
        Noise movie
    """
    im *= 255
    ret = im.astype(np.uint8)
    return ret

def generate_frames(xsize, ysize, tsize, timepoints, levels=10, xyscale=.5, tscale=1, xscale=1.0, yscale=1.0, fps=30, seed=0):
    """Preprocess arguments before passing to the C implementation of Perlin noise.
    """
    # Use the temporal scale and number of timepoints to compute how many
    # units to make the stimulus across the temporal dimension
    tunits = int(tsize/(tscale*(fps/30)))
    if tunits >= 4096:
        raise ValueError("Too many time points.  Either make the tscale larger or tsize smaller")
    ts_all = np.arange(0, tsize, dtype="float32")/(tscale*(fps/30))
    ratio = int(xsize/ysize*XYSCALEBASE)
    arr = _perlin.make_perlin(np.arange(0, xsize, dtype="float32")/ysize/xscale, # Yes, divide by y size
                              np.arange(0, ysize, dtype="float32")/ysize/yscale,
                              ts_all[timepoints],
                              octaves=levels,
                              persistence=xyscale,
                              repeatx=ratio,
                              repeaty=XYSCALEBASE,
                              repeatz=tunits,
                              base=seed)
    arr = arr.swapaxes(0,1)
    return arr
