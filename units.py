from Bio import SeqIO
from tqdm import tqdm
import os

import torch
from transformers import AutoTokenizer, AutoModel
from esm import FastaBatchedDataset, pretrained

def split_fasta_chunk(input_file, output_file, chunk_size):
    with open(output_file, 'w') as out_handle:
        for record in SeqIO.parse(input_file, 'fasta'):
            sequence = record.seq
            seq_length = len(sequence)
            
            num_chunks = -(-seq_length // chunk_size)
            total_chunk_length = num_chunks * chunk_size
            repeat_length = total_chunk_length - seq_length
            repeat_region = repeat_length / num_chunks
            lower_int = int(repeat_region)
            upper_int = lower_int + 1
            
            low_up_numbers = [lower_int] * (num_chunks - 1)
            total_low_up_numbers = sum(low_up_numbers)
            need_to_add_up_nums = repeat_length - total_low_up_numbers
            final_low_up_numbers = [upper_int] * (need_to_add_up_nums) + [lower_int] * (num_chunks - 1 - (need_to_add_up_nums)) + [0]
            move_step = 0
            for a, b in zip(range(0, seq_length, chunk_size), final_low_up_numbers):
                if a > 1:
                    chunk = record[a-move_step:a-move_step + chunk_size]
                else:
                    chunk = record[a:a + chunk_size]
                new_record = chunk
                new_record.id = f"{record.id}_chunk_{a // chunk_size + 1}"
                new_record.description = ""
                SeqIO.write(new_record, out_handle, 'fasta')
                move_step += b

def create_refseq_pro_list(gbff_file):
    gbff = SeqIO.parse(gbff_file, "genbank")
    refseq_pro_dir = {}
    for record in gbff:
        protein_list = []
        for feature in record.features:
            if feature.type == "CDS":
                protein_seq = feature.qualifiers.get("translation", [""])[0]
                protein_list.append(protein_seq)
        refseq_pro_dir[record.id] = protein_list
    return refseq_pro_dir

def reverse_complement(sequence):
        complement = str.maketrans('ACGTacgt', 'TGCAtgca')
        return sequence.translate(complement)[::-1]

def translate_dna(sequence):
    codon_table = {
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S', 'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': '*', 'TAG': '*', 'TGC': 'C', 'TGT': 'C', 'TGA': '*', 'TGG': 'W',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L', 'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q', 'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M', 'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K', 'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V', 'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E', 'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    }

    def translate_frame(sequence, frame):
        return ''.join(codon_table.get(sequence[i:i+3], '') for i in range(frame, len(sequence)-2, 3))
    
    frames = [translate_frame(sequence, frame) for frame in range(3)]
    rev_comp_sequence = reverse_complement(sequence)
    frames += [translate_frame(rev_comp_sequence, frame) for frame in range(3)]
    
    return frames

def seq_in_reflist(sequence, refseq_pro_list):
    if any(sequence in refpro for refpro in refseq_pro_list):
        return True
    else:
        return False

def identify_seq(seqid, sequence, refseq_pro_list = None, istraindata = False):
    final_list = []
    if istraindata:
        ambiguous_bases = {'N', 'R', 'Y', 'M', 'K', 'S', 'W', 'H', 'B', 'V', 'D'}
        selected_seq_pro_dir = {'seqid': seqid, 'nucleotide': sequence, 'protein': ''}
        if any(base in sequence for base in ambiguous_bases):
            return None
        else:
            proteins_list = translate_dna(sequence)
            for num, item in enumerate(proteins_list, start=1):
                if istraindata:
                    if "*" not in item and seq_in_reflist(item, refseq_pro_list) == True:
                        pro_len = len(item)
                        seq_len = pro_len * 3
                        strand = "+" if num <= 3 else "-"
                        if strand == "+":
                            selected_seq = sequence[num-1:num+seq_len]
                            seqid_final = seqid + 'F' + str(num)
                        else:
                            selected_seq = reverse_complement(sequence)[num-3-1:num-3+seq_len]
                            seqid_final = seqid + 'R' + str(num - 3)
                        selected_seq_pro_dir = {
                            'seqid': seqid_final,
                            'nucleotide': selected_seq,
                            'protein': item
                        }
                        final_list.append(selected_seq_pro_dir)
                else:
                    if "*" not in item:
                        pro_len = len(item)
                        seq_len = pro_len * 3
                        strand = "+" if num <= 3 else "-"
                        if strand == "+":
                            selected_seq = sequence[num-1:num+seq_len]
                            seqid_final = seqid + 'F' + str(num)
                        else:
                            selected_seq = reverse_complement(sequence)[num-3-1:num-3+seq_len]
                            seqid_final = seqid + 'R' + str(num - 3)
                        selected_seq_pro_dir = {
                            'seqid': seqid_final,
                            'nucleotide': selected_seq,
                            'protein': item
                        }
                        final_list.append(selected_seq_pro_dir)
            return final_list
    else:
        proteins_list = translate_dna(sequence)
        selected_seq_pro_dir = {'seqid': seqid, 'nucleotide': sequence, 'protein': ''}
        for num, item in enumerate(proteins_list, start=1):
            if "*" not in item:
                pro_len = len(item)
                seq_len = pro_len * 3
                strand = "+" if num <= 3 else "-"
                if strand == "+":
                    selected_seq = sequence[num-1:num+seq_len]
                    seqid_final = seqid + 'F' + str(num)
                else:
                    selected_seq = reverse_complement(sequence)[num-3-1:num-3+seq_len]
                    seqid_final = seqid + 'R' + str(num - 3)
                selected_seq_pro_dir = {
                    'seqid': seqid_final,
                    'nucleotide': selected_seq,
                    'protein': item
                }
                final_list.append(selected_seq_pro_dir)
        return final_list



def process_record(index, record, refseq_pro_list = None, istraindata = False):
    sequence = str(record.seq).upper()
    seqid = record.id
    if istraindata:
        result = identify_seq(seqid, sequence, refseq_pro_list, istraindata)
    else:
        result = identify_seq(seqid, sequence)
    return (index, result)


def extract_DNABERT_S(input_file, out_file,
                      model_loaded = False, tokenizer = None, model = None):

    if model_loaded == False:
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
        model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
    else:
        tokenizer = tokenizer
        model = model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    model.to(device)
    model.eval()

    data = []
    nucleotide = []

    records = list(SeqIO.parse(input_file, 'fasta'))
    for record in tqdm(records, desc="Processing sequences"):
        seq = str(record.seq)
        label = record.id

        inputs = tokenizer(seq, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            
            hidden_states = model(inputs["input_ids"])[0]
            embedding_mean = torch.mean(hidden_states, dim=1)

        result = {
            "label": label,
            "mean_representation": embedding_mean.squeeze().cpu().tolist()
        }
        nucleotide.append(label)
        data.append(result)

    if out_file is not None:
        torch.save({'nucleotide': nucleotide, 'data': data}, out_file)


def extract_esm(fasta_file, 
                model_location='esm2_t36_3B_UR50D',
                truncation_seq_length=1024, toks_per_batch=2048,
                out_file=None, model_loaded = False, model = None, alphabet = None):
    if out_file is not None and os.path.exists(out_file):
        obj = torch.load(out_file)
        data = obj['data']
        proteins = obj['proteins']
        return proteins, data
    if model_loaded == False:
        model, alphabet = pretrained.load_model_and_alphabet(model_location)
    else:
        model = model
        alphabet = alphabet
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = FastaBatchedDataset.from_file(fasta_file)
        
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )
    print(f"Read {fasta_file} with {len(dataset)} sequences")

    return_contacts = False

    repr_layers = [36,]

    proteins = []
    data = []

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):

            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")

            if device:
                toks = toks.to(device, non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

            representations = {
                layer: t.to(device) for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):

                result = {"label": label, "mean_representations": {}}
                truncate_len = min(truncation_seq_length, len(strs[i]))
                result["mean_representations"] = {
                    layer: t[i, 1 : truncate_len + 1].mean(0).clone().to('cpu')
                    for layer, t in representations.items()
                }
                proteins.append(label)
                data.append(result["mean_representations"][36])

    if out_file is not None:
        torch.save({'proteins': proteins, 'data': data}, out_file)
    return proteins, data

def split_fasta_file(input_file, output_dir, sequences_per_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_count = 0
    sequence_count = 0
    current_output_file = None

    with open(input_file, 'r') as infile:
        for line in infile:
            if line.startswith('>'):
                if sequence_count % sequences_per_file == 0:
                    if current_output_file:
                        current_output_file.close()
                    file_count += 1
                    current_output_file = open(os.path.join(output_dir, f'output_{file_count}.fa'), 'w')
                sequence_count += 1
            if current_output_file:
                current_output_file.write(line)

    if current_output_file:
        current_output_file.close()

def merge_data(DNABERT_S_data, ESM_data, merged_file, data_type = None):
    
    DNABERT_S_outfile = torch.load(DNABERT_S_data)
    ESM_outfile = torch.load(ESM_data)

    
    nucleotide_data_dict = {}
    protein_data_dict = {}

    merged_data = []

    for nucleotide, data in zip(DNABERT_S_outfile['nucleotide'], DNABERT_S_outfile['data']):
        nucleotide_data_dict[nucleotide] = torch.tensor(data['mean_representation'])
    for protein, data in zip(ESM_outfile['proteins'], ESM_outfile['data']):
        protein_data_dict[protein] = data
    
    for item in DNABERT_S_outfile['nucleotide']:
        if item in protein_data_dict and item in nucleotide_data_dict:
            protein_data = protein_data_dict[item]
            nucleotide_data = nucleotide_data_dict[item]

            merged_feature = torch.cat((nucleotide_data, protein_data), dim=-1)
            merged_data.append(merged_feature)
        else:
            print(f"Warning: {item} not found in both datasets")

    merged_data = torch.stack(merged_data)
    if data_type == 'host':
        merged_torch = {'ids': DNABERT_S_outfile['nucleotide'], 'data': merged_data, 'labels': [0]}
    elif data_type == 'viral':
        merged_torch = {'ids': DNABERT_S_outfile['nucleotide'], 'data': merged_data, 'labels': [1]}
    elif data_type == None:
        merged_torch = {'ids': DNABERT_S_outfile['nucleotide'], 'data': merged_data}
    torch.save(merged_torch, merged_file)
