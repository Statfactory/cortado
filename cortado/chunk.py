import cortado.seq
import numpy as np

def read_binary(path, length, dtype):
    with open(path, "rb") as iostream:
        buf = np.empty(length, dtype = dtype)
        mem_view = memoryview(buf)
        n = iostream.readinto(mem_view)
        return buf

def next_slice_indices(start_len_slicelen):
    start, length, slicelen = start_len_slicelen
    slicelen = min(length, slicelen)
    newstart = start + slicelen
    newlen = length - slicelen
    if slicelen <= 0:
        return None, start_len_slicelen
    else:
        return (start, start + slicelen), (newstart, newlen, slicelen)

def next_data_chunk(state):
    buf, iostream, start, length, slicelen = state
    slicelen = min(length, slicelen) 
    itemsize = buf.dtype.itemsize
    
    if slicelen <= 0:
        iostream.close()
        return None, None
    else:
        if slicelen < len(buf):
            buf = buf[:slicelen]
        mem_view = memoryview(buf)
        n = iostream.readinto(mem_view)
        
        if n == 0:
            iostream.close()
            return None, None
        elif n // itemsize == slicelen:
            return buf, (buf, iostream, start + slicelen, length - slicelen, slicelen)
        else:
            buf = buf[:(n // itemsize)]
            return buf, (buf, iostream, start + (n // itemsize), length - (n // itemsize), slicelen)