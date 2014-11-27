# Unpublished Copyright (c) 2014 Jeffrey Weiner, All Rights Reserved.
#
# NOTICE: All information contained herein is, and remains the property of Jeffrey
# Weiner. The intellectual and technical concepts contained herein are
# proprietary to Jeffrey Weiner and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material is strictly
# forbidden unless prior written permission is obtained from Jeffrey Weiner.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure of this source code, which includes  information
# that is confidential and/or proprietary, and is a trade secret, of Jeffrey
# Weiner. ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE,
# OR PUBLIC DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS
# WRITTEN CONSENT OF JEFFREY WEINER IS STRICTLY PROHIBITED, AND IN VIOLATION OF
# APPLICABLE LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS
# SOURCE CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
# SELL ANYTHING THAT IT MAY DESCRIBE, IN WHOLE OR IN PART.

from collections import Counter
import math

import numpy as np

def _split(data):
    res = []
    freqs = data[:,1]
    indices = np.where(freqs[0:] - np.concatenate(([0], freqs[:-1])) < 0)[0]
    for i in range(len(indices)):
        start = indices[i - 1] if i != 0 else 0
        end = indices[i]
        res.append(data[start:end])
    res.append(data[indices[-1]:])
    return tuple(res)

def _volts_squared_from_dbm(data):
    IMPEDANCE = 50
    return np.concatenate((data[:,:4], 
                          (10**(data[:,4:]/20.0 - 3) * IMPEDANCE)),
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

def _overlapped(data, percent_overlap):
    num_rows = len(data)
    num_cols = len(data[0,4:])
    num_overlap = int(round(num_cols * percent_overlap))
    num_middle = num_cols - num_overlap * 2
    
    if num_middle < 0:
        raise ValueError('The percent overlap is too large (> 50%)')

    output = np.zeros(num_rows * (num_cols - num_overlap) + num_overlap + 3)
    output[0] = np.mean(data[:,0])
    output[1] = data[0,1]
    output[2] = np.mean(data[:,2])

    i = 3
    data = data[:,4:]
    for r in range(len(data)):
        beg, mid, end = data[r][:num_overlap], \
                          data[r][num_overlap:-num_overlap], \
                          data[r][-num_overlap:]

        # Add beginning if first row
        if r == 0:
            output[i:i + len(beg)] = beg
            i += len(beg)

        # Add middle, always
        output[i:i + len(mid)] = mid
        i += len(mid)

        # Add end if last row, otherwise do overlapping
        if r == len(data) - 1:
            output[i:i + len(end)] = end
        else:
            next_beg = data[r + 1][:num_overlap]
            overlap = np.mean((np.vstack([end, next_beg])), axis=0)
            output[i:i + len(end)] = overlap
        i += len(end)

    return output

def fullsweeper(data, percent_overlap):

    # Step 1: Split the data into fullsweeps
    fullsweeps = _split(data)

    # Step 2: Remove the first and last fullsweep
    if len(fullsweeps) < 3:
        raise ValueError('The data must contain at least one full fullsweep.')
    fullsweeps = fullsweeps[1:-1]

    # Step 3: Convert everything to volts^2
    fullsweeps = [_volts_squared_from_dbm(x) for x in fullsweeps]

    # Step 4: Stack rows with the same startfreq
    fullsweeps = [_stacked(x) for x in fullsweeps]

    # Step 5: Overlap rows based on the percent overlap
    fullsweeps = [_overlapped(x, percent_overlap) for x in fullsweeps]

    # Step 6: Remove any output rows that have a different length from the rest
    mode_length = Counter([len(x) for x in fullsweeps]).most_common(1)[0][0]
    fullsweeps = [x for x in fullsweeps if len(x) == mode_length]

    # Step 7: Stack the remaining output rows into a 2D array and return
    return np.vstack(fullsweeps)
