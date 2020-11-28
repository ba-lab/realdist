#!/bin/bash

set -e

[ $# -ne 3 ] && { echo "Usage: $0 <model> <fasta> <outdir>"; exit 1; }

rootdir=`dirname "$0"`
MODELREALDIST=$1
infasta=$2
outdir=$3
fastaid=`basename "$infasta" .fasta`

echo ""
echo $MODELREALDIST
echo $infasta
echo $outdir
echo $fastaid

SPLITFASTA=$rootdir/split-seq.py
MERGEDMAP=$rootdir/merge-dmaps.py
REALDISTPRED=$rootdir/predict.py
BUILDMSA=/ssdA/common-tools/DeepMSA/hhsuite2/scripts/build_MSA.py

HHBDB="/ssdA/common-tools/uniclust30_2018_08_hhsuite/uniclust30_2018_08"
JACKDB="/ssdA/common-tools/uniref_2019_10_24/uniref90.fasta"
HMMDB="/ssdA/common-tools/mgy_clusters_2018_04/emg_peptides.fa:/ssdA/common-tools/mgy_clusters_2019_05/mgy_clusters.fa:/ssdA/common-tools/fasta_188GB_plass_soedinglab/SRC.fasta:/ssdA/common-tools/metaclust-2018-Jun-22/metaclust_nr.fasta"

mkdir -p $outdir

echo ""
echo "Split fasta"
python3 $SPLITFASTA $infasta $outdir

echo ""
echo "Will run DeepMSA for the following fasta files:"
for fasta in $outdir/*.fasta; do
    echo $fasta
done

mkdir -p $outdir/deepmsa

for fasta in $outdir/$fastaid*.fasta; do
    id=$(basename $fasta .fasta)
    echo ""
    alnfile=$outdir/deepmsa/$id/$id.aln
    echo "Checking for $alnfile"
    if [[ -f $alnfile ]]
    then
        echo "DeepMSA already done!"
    else
        echo "Launching DeepMSA for $fasta"
        set -x
        $BUILDMSA $fasta -outdir=$outdir/deepmsa/$id/ -hhblitsdb=$HHBDB -jackhmmerdb=$JACKDB -hmmsearchdb=$HMMDB &> $outdir/deepmsa-$id.log
        set +x
    fi
done

echo ""
echo "Will run Distance prediction for the following MSA files:"
for aln in $outdir/deepmsa/*/$fastaid*.aln; do
    wc -l $aln
done

for aln in $outdir/deepmsa/*/*.aln; do
    id=$(basename $aln .aln)
    echo ""
    echo $id
    echo $aln
    if [[ -f $outdir/msa2dist/$id.realdist.msa.npy ]]
    then
        echo "MSA Distance prediction already done!"
    else
        echo "Launching MSA distance prediction for $aln"
        set -x
        CUDA_VISIBLE_DEVICES="" python3 $REALDISTPRED -w $MODELREALDIST -a $aln -o $outdir/msa2dist/ &> $outdir/msa2dist-$id.log
        set +x
    fi
done

echo ""
echo "Merge distance map predictions.."
python3 $MERGEDMAP $outdir/$fastaid.dict $outdir/msa2dist/ $outdir/
