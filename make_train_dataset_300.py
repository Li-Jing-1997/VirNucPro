from units import *
import time
import gzip
from tqdm import tqdm
def process_record(record, refseq_pro_dir):
    sequence = str(record.seq).upper()
    seqid = record.id
    refseq_pro_key = seqid.split('_chunk_')[0]
    refseq_pro_list = refseq_pro_dir.get(refseq_pro_key, [])
    result = identify_seq(seqid, sequence, refseq_pro_list, istraindata=True)
    return result

def process_file(gbff_file, chunk_file):

    identified_nucleotide_file = chunk_file.replace('genomic.chunk300.fna', 'identified_nucleotide.fa')
    identified_protein_file = chunk_file.replace('genomic.chunk300.fna', 'identified_protein.fa')
    if os.path.exists(identified_nucleotide_file) and os.path.exists(identified_protein_file):
        print(f"Both {identified_nucleotide_file} and {identified_protein_file} exist, skipping...")
        return
    start_time = time.time()
    refseq_pro_dir = create_refseq_pro_list(gbff_file)
    print(f"Read {gbff_file} done, time taken: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    records = list(SeqIO.parse(chunk_file, 'fasta'))
    print(f"Read {chunk_file} done, time taken: {time.time() - start_time:.2f} seconds")

    with open(identified_nucleotide_file, 'w') as dna_out, open(identified_protein_file, 'w') as protein_out:
        start_time = time.time()
        for record in tqdm(records, total=len(records)):
            result = process_record(record, refseq_pro_dir)
            if result:
                for item in result:
                    if item.get('protein', '') != '':
                        sequence_name = item['seqid']
                        dna_sequence = item['nucleotide']
                        protein_sequence = item['protein']
                        
                        dna_out.write(f'>{sequence_name}\n')
                        dna_out.write(f'{dna_sequence}\n')
                        
                        protein_out.write(f'>{sequence_name}\n')
                        protein_out.write(f'{protein_sequence}\n')
        print(f"write nucleotide and protein done, time taken: {time.time() - start_time:.2f} seconds")

def main():
    fna_list = []
    gbff_list = []
    chunk_size = 300

    for root, dirs, filenames in os.walk('./data'):
        for filename in filenames:
            if filename.endswith('genomic.fna.gz'):
                fna_list.append(os.path.join(root, filename))
            elif filename.endswith('genomic.gbff.gz'):
                gbff_list.append(os.path.join(root, filename))

    for gbff_file_gz in gbff_list:
        base_name = gbff_file_gz.replace('genomic.gbff.gz', '')
        for infile_gz in fna_list:
            if infile_gz.startswith(base_name):
                output_file = infile_gz.replace('genomic.fna.gz', 'genomic.chunk300.fna')
                with gzip.open(infile_gz, 'rt') as f_in:
                    infile = infile_gz.replace('genomic.fna.gz', 'genomic.fna')
                    with open(infile, 'w') as f_out:
                        f_out.write(f_in.read())
                with gzip.open(gbff_file_gz, 'rt') as f_in:
                    gbff_file = gbff_file_gz.replace("genomic.gbff.gz", 'genomic.gbff')
                    with open(gbff_file, 'w') as f_out:
                        f_out.write(f_in.read())
                split_fasta_chunk(infile, output_file, chunk_size)
                os.remove(infile)
                process_file(gbff_file, output_file)
                os.remove(gbff_file)
                os.remove(output_file)
if __name__ == "__main__":
    main()
