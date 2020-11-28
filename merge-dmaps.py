'''
Author: Badri Adhikari, University of Missouri-St. Louis,  03-29-2020
File: Merge chopped distance maps to a full distance using np.nanmin()
'''

import sys, os
import numpy as np
import matplotlib.pyplot as plt

splitsdict = sys.argv[1]
dmapfolder = sys.argv[2]
outfolder  = sys.argv[3]

id = os.path.splitext(os.path.basename(splitsdict))[0]

f = open(splitsdict)
subsequences = eval(f.readline().strip())
f.close()

# Identify the full sequence key/value entry
myrange = 0
mainkey = ''
mainseq = ''
for i, j in subsequences.items():
    a, b = i.split('-')
    if int(b) - int(a) > myrange:
        myrange = int(b) - int(a)
        mainkey = i
        mainseq = j
del subsequences[mainkey]

dmap = dmapfolder + '/' + id + '.' + mainkey + '.realdist.msa.npy'
print(f'Loading {dmap}')
A = np.load(dmap)
L = len(A[:, 0])
print(A.shape, L)

for i, j in subsequences.items():
    a, b = i.split('-')
    a = int(a) - 1
    b = int(b)
    B = np.full((L, L), np.nan)
    dmap = dmapfolder + '/' + id + '.' + i + '.realdist.msa.npy'
    print('')
    print(f'Loading {dmap} => {a} - {b}')
    x = np.load(dmap)
    print(x.shape)
    B[a:b, a:b] = x
    MIN = np.nanmin(np.stack((A, B), axis = -1), axis = -1)
    A = MIN

A[A>30] = 30

rr = open(outfolder + '/' + id + '.rr', 'w')
rr.write(mainseq + '\n')
for j in range(0, L):
    for k in range(j, L):
        if abs(j - k) < 5:
            continue
        rr.write("%i %i 0 8 %.5f # d = %.2f\n" %(j+1, k+1, 4.0 / A[j][k], A[j][k]) )
rr.close()

np.save(outfolder + '/' + id + '.realdist.npy', A)

plt.imshow(A, cmap='Spectral')
plt.title(id)
plt.colorbar()
plt.savefig(outfolder + '/' + id + '.realdist.npy' + '.png')

print('')
print(f'Written rr and dmap!')

