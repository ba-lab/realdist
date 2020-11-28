'''
Author: Badri Adhikari, University of Missouri-St. Louis,  12-28-2019
File: Generate features for a fasta input
'''

from datetime import datetime
import os
import sys
import numpy as np
import subprocess
import pickle

################################################################################
if sys.version_info < (3,0,0):
    print('Python 3 required!!!')
    sys.exit(1)

################################################################################
alnfile = sys.argv[1]
jobdir = sys.argv[2]
outfile = sys.argv[3]

################################################################################
if len(alnfile) < 2:
    print('alnfile undefined!')
    usage()
    sys.exit()

if len(jobdir) < 2:
    print('jobdir undefined!')
    usage()
    sys.exit()

if len(outfile) < 2:
    print('outfile undefined!')
    usage()
    sys.exit()

################################################################################
start = datetime.now()

if not os.path.exists(jobdir):
    os.system('mkdir -p ' + jobdir)

id = os.path.splitext(os.path.basename(alnfile))[0]
os.chdir(jobdir)

################################################################################
ALNSTAT  = '/ssdA/common-tools/metapsicov/metapsicov-master/bin/alnstats'
CCMPRED  = '/ssdA/common-tools/ccmpred/CCMpred/bin/ccmpred'
FREECON  = '/usr/bin/freecontact'

################################################################################
if not os.path.exists(ALNSTAT):
    sys.exit(ALNSTAT + ' not found')
if not os.path.exists(CCMPRED):
    sys.exit(CCMPRED + ' not found')
if not os.path.exists(FREECON):
    sys.exit(FREECON + ' not found')

if not os.path.exists(alnfile):
    print('ERROR! file not found!')
    print(alnfile)
    sys.exit()

################################################################################
print('Started ' + sys.argv[0] + ' ' + str(start))
print('ID: ' + id)
print('Outdir: ' + jobdir)
print('Alnfile: ' + alnfile)
f = open(alnfile, 'r')
seq = f.readline().strip()
print('L: ' + str(len(seq)))
print('Seq: ' + seq)
sys.stdout.flush()

fw = open(id + '.fasta', 'w')
fw.write('>' + id + '\n')
fw.write(seq + '\n')
fw.close()
fasta = id + '.fasta'

################################################################################
fin = open(alnfile, 'r')
lines = fin.readlines()
fin.close()

if seq != lines[0].strip():
    print('ERROR! Fasta seq and aln seq dont match')
    print(seq)
    print(lines[0])
    sys.exit()

print('')
os.system('wc -l ' + id + '.aln')

################################################################################
print ('')
print ('Run alnstats..')
sys.stdout.flush()
if os.path.exists(id + '.colstats') and os.path.exists(id + '.pairstats'):
    print('alnstats already done!')
else:
    response = subprocess.run([ALNSTAT, id + '.aln', id + '.colstats', id + '.pairstats'])
    if (response.returncode != 0):
        sys.exit(str(response.returncode) + ' ' + str(response.stderr) + ' ' + str(response.args))

################################################################################
print ('')
print ('Run CCMpred..')
sys.stdout.flush()
if os.path.exists(id + '.ccmpred'):
    print('ALN already done!')
else:
    ccm_outfile = open('ccmpred.log', 'w')
    response = subprocess.run([CCMPRED, id + '.aln', id + '.ccmpred'], stdout = ccm_outfile)
    ccm_outfile.close()
    if (response.returncode != 0):
        sys.exit(str(response.returncode) + ' ' + str(response.stderr) + ' ' + str(response.args))

################################################################################
print ('')
print ('Run FreeContact..')
sys.stdout.flush()
if os.path.exists(id + '.freecontact.rr'):
    print('ALN already done!')
else:
    if os.system(FREECON + ' < ' + id + '.aln' + ' > ' + id + '.freecontact.rr') != 0:
        faln = open(id + '.aln', 'r')
        lines_aln = faln.readlines()
        faln.close()
        if len(lines_aln) > 5:
            sys.exit(FREECON + 'Failed!')
        else:
            print('WARNING! Aln is too small, freecontact failed!')
            os.system('touch ' + id + '.freecontact.rr')

################################################################################
# Check the presence of all files
assert os.path.exists(id + '.aln')
assert os.path.exists(id + '.ccmpred')
assert os.path.exists(id + '.freecontact.rr')

################################################################################
print('')
print('Write feature file..')

# Add sequence
features = {'seq': seq }

# Add CCMpred
f = open(id + '.ccmpred', 'r')
ccmlist = [[float(num) for num in line.strip().split()] for line in f ]
ccmpred = np.array(ccmlist)
assert ccmpred.shape == (( len(seq), len(seq) ))
features['ccmpred'] = ccmpred

# Add FreeContact
freecontact = np.zeros((len(seq), len(seq)))
f = open(id + '.freecontact.rr', 'r')
for l in f.readlines():
    c = l.strip().split()
    #assert c[1] == seq[int(c[0]) - 1]
    #assert c[3] == seq[int(c[2]) - 1]
    freecontact[int(c[0]) - 1, int(c[2]) - 1] = float(c[5])
    freecontact[int(c[2]) - 1, int(c[0]) - 1] = float(c[5])
features['freecon'] = freecontact

# Add Shanon entropy
entropy = []
f = open(id + '.colstats', 'r')
f.readline()
f.readline()
f.readline()
f.readline()
for l in f.readlines():
    c = l.strip().split()
    entropy.append(c[21])
assert len(entropy) == len(seq)
features['entropy'] = np.array(entropy)

# Add mean contact potential
potential = np.zeros((len(seq), len(seq)))
f = open(id + '.pairstats', 'r')
for l in f.readlines():
    c = l.strip().split()
    potential[int(c[0]) - 1, int(c[1]) - 1] = float(c[2])
    potential[int(c[1]) - 1, int(c[0]) - 1] = float(c[2])
features['potential'] = potential

################################################################################
pickle.dump(features, open(outfile, 'wb'))

print('done ' + sys.argv[0] + ' ' + str(datetime.now()))
################################################################################
