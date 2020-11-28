'''
Author: Badri Adhikari, University of Missouri-St. Louis,  03-29-2020
File: Splits fasta file into X-sized chopped fasta files
Given CROP = 192, Outputs will be as follows:
1-192, 97-288, 193-384, 289-480, 385-576, 481- ...
'''

import sys, os

CROP = 256
FLEX = 32

filefasta = sys.argv[1]
outdir = os.path.abspath(sys.argv[2])

id = os.path.splitext(os.path.basename(filefasta))[0]

f = open(filefasta)
f.readline()
sequence = f.readline().strip()
f.close()

L = len(sequence)

subsequences = {str(1) + '-' + str(L): sequence}
if L > CROP + FLEX:
    start = 0
    while(True):
        if start + CROP + FLEX > L:
            if L - start < CROP:
                start = L - CROP
            subsequences[str(start + 1) + '-' + str(L)] = sequence[start:]
            break
        subsequences[str(start + 1) + '-' + str(start+CROP)] = sequence[start:start+CROP]
        start += int(CROP/2)

for i, seq in subsequences.items():
    f = open(outdir + '/' + id + '.' + i + '.fasta', 'w')
    f.write('>' + id + ' ' + i + '\n')
    print(i)
    f.write(seq + '\n')
    print(seq)
    f.close()

f = open(outdir + '/' + id + '.dict', 'w')
f.write(str(subsequences))
f.close()
