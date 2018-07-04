# Get the source directory to call the python script further
SRC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the current directory
THISDIR=`pwd`

# Install biopython, useful to deal with the .sam files
python -c "from Bio import SeqIO"
EXIT_STATUS=$?
if [ ! "$EXIT_STATUS" -eq "0" ]; then
  conda install biopython
  EXIT_STATUS=$?
  if [ ! "$EXIT_STATUS" -eq "0" ]; then
    pip install biopython
  fi
fi

# Download E. coli ONT data and reference genome
cd $SRC_DIR
if [ ! -d ecoli_data ]; then
  mkdir ecoli_data;
fi
cd ecoli_data
if [ ! -f reads.fasta ]; then
  wget https://nanopore.s3.climb.ac.uk/MAP006-PCR-1_2D_pass.fasta
  mv MAP006-PCR-1_2D_pass.fasta reads.fasta
fi
if [ ! -d ref ]; then
mkdir ref;
fi
cd ref
if [ ! -f ref.fna ]; then
  wget ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz
  gunzip GCF_000005845.2_ASM584v2_genomic.fna.gz
  mv GCF_000005845.2_ASM584v2_genomic.fna ref.fna
fi
cd ../../

# Download minimap2 if it is not on your path
# minimap2 is a tool that computes the overlaps between reads
# and maps the reads to a reference genome to know their position (ground truth for the ordering)
MINIMAP=`command -v minimap2`
if [ "${#MINIMAP}" -eq 0 ]; then  # check if minimap is on your path
    if [ ! -d software ]; then  # create a software directory if it does not exist
        mkdir software;
    fi
    cd software
    if [ ! -d minimap2 ]; then
      git clone https://github.com/lh3/minimap2  # clone minimap2
      cd minimap2 && make  # must have dependencies installed for minimap2, such as zlib.
      cd ../
    fi
    cd ../
    MINIMAP=${THISDIR}/software/minimap2/minimap2
fi

# if [ ! -d results ]; then  # create a directory if it does not exist
#     mkdir results;
# fi

# Run minimap to get the read overlaps
$MINIMAP -x ava-ont ${SRC_DIR}/ecoli_data/reads.fasta ${SRC_DIR}/ecoli_data/reads.fasta > ${SRC_DIR}/ecoli_data/ovlp.paf

# Run minimap2 to get the ground truth position of the reads on the genome
$MINIMAP -ax map-ont ${SRC_DIR}/ecoli_data/ref/ref.fna ${SRC_DIR}/ecoli_data/reads.fasta > ${SRC_DIR}/ecoli_data/aln.sam

# Get the positions of the reads on the genome from the .sam file
# python $DIR/get_pos_from_sam.py results/aln.sam ecoli_data/reads.fasta -o $DIR/results/reads_pos.csv

# Run multidimensional spectral ordering in python on this data
