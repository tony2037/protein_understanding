from Bio import SeqIO
import os, sys
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("source", help="source fasta file")
parser.add_argument("output", help="output pretrain file")
parser.add_argument("-l", "--limit-length", help="limit length", dest="length", default=512)
args = parser.parse_args()
fasta_file = args.source
output_path = args.output
limit_length = int(args.length)
print("source file:", fasta_file)
print("output file:", output_path)
print("limit length:", limit_length)

sequences = []
for s in SeqIO.parse(fasta_file,"fasta"):
    tmp = str(s.seq).upper()
    if len(tmp) > limit_length:
        continue
    seq = ' '.join(tmp)
    ans = '{}\t1'.format(seq, seq)
    sequences.append(ans)

with open(output_path, 'w') as f:
    f.write('sentence\tlabel\n')
    ans = '\n'.join(sequences)
    f.write(ans)
