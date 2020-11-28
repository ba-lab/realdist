#!/bin/bash

set -e

[ $# -ne 4 ] && { echo "Usage: $0 <model> <fasta> <aln> <outdir>"; exit 1; }

rootdir=`dirname "$0"`
MODELREALDIST=$1
infasta=$2
inaln=$3
outdir=$4
fastaid=`basename "$infasta" .fasta`

echo ""
echo $infasta
echo $inaln
echo $outdir
echo $fastaid

REALDISTPRED=$rootdir/predict.py

mkdir -p $outdir

echo ""

if [[ -f $outdir/$fastaid.realdist.msa.npy ]]
then
    echo "MSA Distance prediction already done!"
else
    echo "Launching MSA distance prediction for $fastaid"
    set -x
    python3 $REALDISTPRED -w $MODELREALDIST -a $inaln -o $outdir/
    set +x
fi
