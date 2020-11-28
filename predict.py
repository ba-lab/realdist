'''
Author: Badri Adhikari, University of Missouri-St. Louis,  12-29-2019
File: Contains the code to predict contacts
'''

import os
import sys
import datetime
import pickle
import getopt
import string
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
epsilon = tf.keras.backend.epsilon()
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input, Convolution2D, Activation, add, Dropout, BatchNormalization
from tensorflow.python.keras.models import Model

if sys.version_info < (3,0,0):
    print('Python 3 required!!!')
    sys.exit(1)

# Allow GPU memory growth
if hasattr(tf, 'GPUOptions'):
    import keras.backend as K
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.tensorflow_backend.set_session(sess)
else:
    # For other GPUs
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

def usage():
    print('Usage:')
    print(sys.argv[0] + ' <-w file_weights> <-a file_aln> <-o jobdir>')

try:
    opts, args = getopt.getopt(sys.argv[1:], "w:a:o:h")
except getopt.GetoptError as err:
    print(err)
    usage()
    sys.exit(2)

wts = ''
aln = ''
jobdir = ''
for o, a in opts:
    if o in ("-h", "--help"):
        usage()
        sys.exit()
    elif o in ("-w"):
        wts = os.path.abspath(a)
    elif o in ("-a"):
        aln = os.path.abspath(a)
    elif o in ("-o"):
        jobdir = os.path.abspath(a)
    else:
        assert False, "Error!! unhandled option!!"

if len(wts) < 2:
    print('wts file undefined!')
    usage()
    sys.exit()
if len(aln) < 2:
    print('in aln undefined!')
    usage()
    sys.exit()
if len(jobdir) < 2:
    print('job dir undefined!')
    usage()
    sys.exit()

pad_size             = 10
expected_n_channels  = 322
OUTL                 = 1024
scripts              = os.path.dirname(os.path.abspath(sys.argv[0]))

def get_sequence(pdb, feature_file):
    features = pickle.load(open(feature_file, 'rb'))
    return features['seq']

# Adapted from trRosetta, but works for fasta as well
# read A3M and convert letters into integers in the 0..20 range
def parse_a3m(seq_line):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    seqs.append(seq_line.rstrip().translate(table))
    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i
    assert msa.max() < 20 # for sequences without gaps
    return msa

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def get_feature(file_pre231, file_pssm, file_seq):
    x_seq = get_feature_seq(file_seq, 47)
    L = len(x_seq[:, 0, 0])
    x_pre231 = get_feature_pre231(file_pre231, L)
    x_pssm = get_feature_trRos_profile(file_pssm, L)
    print('All shapes : ', 'seq', x_seq.shape, 'pre231', x_pre231.shape, 'pssm', x_pssm.shape)
    assert len(x_seq[0, 0, :])    == 47
    assert len(x_pre231[0, 0, :]) == 231
    assert len(x_pssm[0, 0, :])   == 44
    X = np.concatenate((x_seq, x_pre231, x_pssm), axis = -1)
    assert X.shape == (L, L, expected_n_channels)
    return X.astype(np.float32)

def get_feature_pre231(file_pre231, L):
    x = np.zeros((L, L, 231), dtype = np.float32)
    if os.path.getsize(file_pre231) > 500:
        x = np.load(file_pre231)
    return x.astype(np.float32)

def get_feature_trRos_profile(file_pssm, L):
    x = np.zeros((L, 22), dtype = np.float32)
    if os.path.getsize(file_pssm) > 500:
        x = np.load(file_pssm)
    X = np.full((L, L, 44), 0.0)
    fi = 0
    for j in range(22):
        a = np.repeat(x[:, j].reshape(1, L), L, axis = 0)
        X[:, :, fi] = a
        fi += 1
        X[:, :, fi] = a.T
        fi += 1
    assert fi == 44
    assert X.max() < 100.0
    assert X.min() > -100.0
    return X

# Adapted from trRosetta, but works for fasta as well
# read A3M and convert letters into integers in the 0..20 range
def parse_a3m(seq_line):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    seqs.append(seq_line.rstrip().translate(table))
    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i
    assert msa.max() < 20 # for sequences without gaps
    return msa

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def get_feature_seq(file_seq, expected_n_channels):
    features = pickle.load(open(file_seq, 'rb'))
    l = len(features['seq'])
    seq = features['seq']
    seq1hot = one_hot(parse_a3m(seq), 20).T
    assert seq1hot.shape == (20, l)
    # Create X and Y placeholders
    X = np.full((l, l, expected_n_channels), 0.0)
    # Add residue index numbers
    fi = 0
    residue_indices = np.asarray([ i/100 for i in range(1, l+1) ])
    X[:, :, fi] = np.repeat(residue_indices.reshape(1, l), l, axis = 0)
    fi += 1
    X[:, :, fi] = X[:, :, fi - 1].T
    fi += 1
    # Add seq 1-hot
    for j in range(20):
        a = np.repeat(seq1hot[j].reshape(1, l), l, axis = 0)
        X[:, :, fi] = a
        fi += 1
        X[:, :, fi] = a.T
        fi += 1
    # Add entrophy
    entropy = features['entropy']
    assert entropy.shape == (l, )
    a = np.repeat(entropy.reshape(1, l), l, axis = 0)
    X[:, :, fi] = a
    fi += 1
    X[:, :, fi] = a.T
    fi += 1
    # Add CCMpred
    ccmpred = features['ccmpred']
    assert ccmpred.shape == ((l, l))
    X[:, :, fi] = ccmpred
    fi += 1
    # Add  FreeContact
    freecon = features['freecon']
    assert freecon.shape == ((l, l))
    X[:, :, fi] = freecon
    fi += 1
    # Add potential
    potential = features['potential']
    assert potential.shape == ((l, l))
    X[:, :, fi] = potential
    fi += 1
    assert fi == expected_n_channels
    assert X.max() < 100.0
    assert X.min() > -100.0
    return X

# Improved DEEPCON for distances
def real_distances(L, num_blocks, intermediate_n_channels, input_n_channels):
    print('')
    print('Model params:')
    print('L', L)
    print('num_blocks', num_blocks)
    print('intermediate_n_channels', intermediate_n_channels)
    print('input_n_channels', input_n_channels)
    print('')
    dropout_value = 0.2
    my_input = Input(shape = (L, L, input_n_channels))
    tower = BatchNormalization()(my_input)
    tower = Activation('elu')(tower)
    tower = Convolution2D(intermediate_n_channels, 1, padding = 'same')(tower)
    flag_1D = False
    d_rate = 1
    for i in range(num_blocks):
        block = BatchNormalization()(tower)
        block = Activation('elu')(block)
        if flag_1D:
            block = Convolution2D(intermediate_n_channels, kernel_size = (1, 5), padding = 'same')(block)
        else:
            block = Convolution2D(intermediate_n_channels, kernel_size = (3, 3), padding = 'same')(block)
        block = Dropout(dropout_value)(block)
        block = Activation('elu')(block)
        if flag_1D:
            block = Convolution2D(intermediate_n_channels, kernel_size = (1, 5), dilation_rate=(d_rate, d_rate), padding = 'same')(block)
            flag_1D = False
        else:
            block = Convolution2D(intermediate_n_channels, kernel_size = (3, 3), dilation_rate=(d_rate, d_rate), padding = 'same')(block)
            flag_1D = True
        tower = add([block, tower])
        if d_rate == 1:
            d_rate = 2
        elif d_rate == 2:
            d_rate = 4
        else:
            d_rate = 1
    tower = BatchNormalization()(tower)
    tower = Activation('relu')(tower)
    tower = Convolution2D(1, 3, padding = 'same')(tower)
    tower = Activation('relu')(tower)
    model = Model(my_input, tower)
    return model

id = os.path.splitext(os.path.basename(aln))[0]
os.system(f"mkdir -p {jobdir}")
os.chdir(jobdir)

faln = open(aln, 'r')
alnlines = faln.readlines()
faln.close()

print('')
print('Make a copy of aln file..')
os.system("egrep -v \"^>\" " + aln + " | sed 's/[a-z]//g' > " + id + ".aln")
aln = id + '.aln'

print('')
print('Generate pre231 feature..')
if not os.path.exists(id + '.pre231.npy'):
    cmd = f"python3 {scripts}/gen-precision231.py /ssdA/common-tools/cov21stats {aln} {id}.pre231.npy"
    if os.system(cmd) != 0:
        sys.exit(cmd)

print('')
print('Generate sequence features..')
if not os.path.exists(id + '.pkl'):
    cmd = f"python3 {scripts}/gen-seq-features.py {aln} ./ {id}.pkl"
    if os.system(cmd) != 0:
        sys.exit(cmd)

print('')
print('Generate profile..')
if not os.path.exists(id + '.pssm.npy'):
    cmd = f"python3 {scripts}/gen-pssm-features.py {aln} {id}.pssm.npy"
    if os.system(cmd) != 0:
        sys.exit(cmd)

print('')
print('Make X..')
X = get_feature(id + '.pre231.npy', id + '.pssm.npy', id + '.pkl')

l = len(X[:, 0, 0])
assert len(X[0, 0, :]) == expected_n_channels
OUTL = l + pad_size
XX = np.full((1, OUTL, OUTL, expected_n_channels), 0.0)
Xpadded = np.zeros((l + pad_size, l + pad_size, len(X[0, 0, :])))
Xpadded[int(pad_size/2) : l+int(pad_size/2), int(pad_size/2) : l+int(pad_size/2), :] = X
l = len(Xpadded[:, 0, 0])
XX[0, :l, :l, :] = Xpadded

print('')
print('Channel summaries:')
print(' Channel        Avg        Max        Sum')
for i in range(len(X[0, 0, :])):
    (m, s, a) = (X[:, :, i].flatten().max(), X[:, :, i].flatten().sum(), X[:, :, i].flatten().mean())
    print(' %7s %10.4f %10.4f %10.1f' % (i+1, a, m, s))

model = real_distances(OUTL, 128, 128, expected_n_channels)

model.load_weights(wts)

P = model.predict(XX)
# Remove padding, i.e. shift up and left by int(pad_size/2)
P[:, :OUTL-pad_size, :OUTL-pad_size, :] = P[:, int(pad_size/2) : OUTL-int(pad_size/2), int(pad_size/2) : OUTL-int(pad_size/2), :]
P[ P < 0.001 ] = 0.001
P = 10.0 / ((P ** (3.0/7.0)) + epsilon)

print('')
print('Save prediction..')

f = open(id + '.aln', 'r')
seq = f.readline().strip()
f.close()

rr = open(id + '.realdist.msa.rr', 'w')
rr.write(seq + "\n")

PP = np.full((len(seq), len(seq)), np.nan)

L = len(seq)

'''
plt.imshow(P[0, :L, :L, 0], cmap='Spectral')
plt.title(id + ' ' + ' using MSA')
plt.colorbar()
plt.savefig(id + '.realdist.msa.rough.' + '.png')
'''

for j in range(0, L):
    for k in range(j, L):
        PP[j, k] = (P[0, k, j, 0] + P[0, j, k, 0]) / 2.0

for j in range(0, L):
    for k in range(j, L):
        if abs(j - k) < 5:
            continue
        rr.write("%i %i 0 8 %.5f # d = %.2f\n" %(j+1, k+1, 4.0 / PP[j][k], PP[j][k]) )
rr.close()
print('Written RR !!')

np.save(id + '.realdist.msa.npy', PP)
print('Written distance map (numpy) !!')

PP[ PP > 30] = 30
plt.imshow(PP, cmap='Spectral')
plt.title(id + ' ' + ' using MSA')
plt.colorbar()
plt.savefig(id + '.realdist.msa' + '.png')

print(f"{sys.argv[0]} done!")
