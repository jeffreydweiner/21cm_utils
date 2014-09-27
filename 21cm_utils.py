import numpy as np

IMPEDANCE = 50

def _fullsweeps(data):
    fullsweeps = []
    cur_fullsweeps = []
    for row in data:
        if cur_fullsweeps and row[1] < cur_fullsweeps[-1][1]:
            fullsweeps.append(cur_fullsweeps)
            cur_fullsweeps = []
        cur_fullsweeps.append(row)
    fullsweeps.append(cur_fullsweeps)
    return [np.vstack(row) for row in fullsweeps]

def _volts_from_dbm(data, impedance):
    return data
    return np.concatenate((data[:,:4], 
                           (10**(data[:,4:]/20.0 - 3) * impedance)),
                          axis=1)

def _stacked(data):
    stacked = []
    curr_stack = []

    def append():
        stacked.append(np.mean(curr_stack, axis=0))
        stacked[-1][3] *= len(curr_stack)

    for row in data:
        if curr_stack and row[1] != curr_stack[-1][1]:
            append()
            curr_stack = []
        curr_stack.append(row)
    append()
    return np.vstack(stacked)

def _split_row(row, num_overlap):
    return row[:num_overlap],             \
           row[num_overlap:-num_overlap], \
           row[-num_overlap:]

def _overlapped(data, percent_overlap):
    num_cols = len(data[0,4:])
    num_overlap = int(round(num_cols * percent_overlap))
    num_middle = num_cols - num_overlap
    if num_middle < 2:
        raise ValueError('percent_overlap is too large')

    output = np.array((np.mean(data[:,0]), data[0,1], np.mean(data[:,2])))
    for r in range(len(data)):
        beg, mid, end = _split_row(data[r][4:], num_overlap)
        if r == 0:
            output = np.concatenate((output, beg))
        output = np.concatenate((output, mid))
        if r == len(data) - 1:
            output = np.concatenate((output, end))
        else:
            next_beg = _split_row(data[r + 1][4:], num_overlap)[0]
            output = np.concatenate((output, 
                                     np.mean((end, next_beg), axis=0)))
    return output

def fullsweeper(data, percent_overlap):
    return np.array(
             [np.ndarray.tolist(
                _overlapped(_stacked(_volts_from_dbm(x,IMPEDANCE)), percent_overlap)) \
             for x in _fullsweeps(data)[1:-1]])