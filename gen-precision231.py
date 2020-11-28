#!/usr/bin/python
# Badri Adhikari, 2/1/2020

import sys, os
import numpy as np
import scipy.linalg as la
import numpy.linalg as na

cov21stats = sys.argv[1]
infile = sys.argv[2]
outpre = sys.argv[3]

tmp21c = ''
if len(sys.argv) > 3:
    tmp21c = sys.argv[3]

# From ResPRE
def ROPE(S, rho):
    p=S.shape[0]
    S=S
    try:
        LM=na.eigh(S)
    except:
        LM=la.eigh(S)
    L=LM[0]
    M=LM[1]
    for i in range(len(L)):
        if L[i]<0:
            L[i]=0
    lamda=2.0/(L+np.sqrt(np.power(L,2)+8*rho))
    indexlamda=np.argsort(-lamda)
    lamda=np.diag(-np.sort(-lamda)[:p])
    hattheta=np.dot(M[:,indexlamda],lamda)
    hattheta=np.dot(hattheta,M[:,indexlamda].transpose())
    return hattheta

print(f"Input: {infile}")

faln = open(infile, 'r')
alnlines = faln.readlines()
faln.close()

seq = ''
for l in alnlines:
    if l.startswith('>'):
        continue
    seq += l.strip()
    break

L = len(seq)

print(f"L: {L}")
print('Rows in aln:')
os.system('wc -l ' + infile)

if not os.path.exists(infile):
    print(f"ERROR!! {infile} does not exist!!")
    exit(1)

if os.path.exists(outpre):
    print(f"{outpre} already exists.. nothing to do..")
    exit(0)

tmp21c = outpre + '.cov.21c'
if os.path.exists(tmp21c):
    print(f"{tmp21c} already exists.. using it..")
else:
    print('Run cov21c..')
    os.system(cov21stats + ' ' + infile + ' ' + tmp21c)

if not os.path.exists(tmp21c):
    print('ERROR!! cov21stats failed!!')
    exit(1)

X = np.zeros((1, L, L, 441))
x_ch_first = np.memmap(tmp21c, dtype=np.float32, mode='r', shape=(1, 441, L, L))
print(x_ch_first.shape)
x = np.rollaxis(x_ch_first[0], 0, 3) # convert to channels_last

L = len(x[:, 0, 0])
print(f"L: {L}")

zd = x.reshape(L, L, 21, 21)
zz = np.moveaxis(zd, 2, 1)
ze = zz.reshape(L * 21, L * 21)

print('Apply ROPE..')
rho2 = np.exp((np.arange(80)-60)/5.0)[30]
zf = ROPE(ze, rho2)

# Postprocess to channels_last
zg = zf.reshape(L, 21, L, 21)
zh = np.moveaxis(zg, 1, 2)
zi = zh.reshape(L, L, 441)

print('Compress..')
compressed_final = np.zeros((L,L,231))
c = 0
for a in range(21):
    for b in range(21):
        if a > b:
            continue
        elif a == b:
            compressed_final[:, :, c] = zi[:, :, 21 * a + b]
        else:
            compressed_final[:, :, c] = zi[:, :, 21 * a + b] + zi[:, :, 21 * b + a]
        c = c + 1

np.save(outpre, compressed_final.reshape(L, L, 231).astype(np.float16))
os.system('rm -f ' + tmp21c)
print(f"{sys.argv[0]} done!")
