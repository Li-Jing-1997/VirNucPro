from units import *
import random
import math
import multiprocessing

nucleotide_input_file_list = []
sequences_per_file = 10000

DNABERT_S_tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
DNABERT_S_model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)

ESM_model, ESM_alphabet = pretrained.load_model_and_alphabet('esm2_t36_3B_UR50D')


def process_file_seq(file):
    output_file = f'{file.split(".fa")[0]}_DNABERT_S.pt'
    merged_file_path = output_file.replace('./data/', './data/data_merge/').replace('.identified_nucleotide', '_merged').replace('DNABERT_S', 'merged')
        
    if os.path.exists(output_file) or os.path.exists(merged_file_path):
        return output_file
    extract_DNABERT_S(input_file=file, out_file=output_file, model_loaded=True, tokenizer=DNABERT_S_tokenizer, model=DNABERT_S_model)
    print('saved to: ' + output_file)
    return output_file

def process_file_pro(file):
    output_file = f'{file.split(".fa")[0]}_ESM.pt'
    merged_file_path = output_file.replace('./data/', './data/data_merge/').replace('.identified_protein', '_merged').replace('ESM', 'merged')

    if os.path.exists(output_file) or os.path.exists(merged_file_path):
        return output_file
    extract_esm(fasta_file = file, out_file = output_file, model_loaded=True, model=ESM_model, alphabet=ESM_alphabet)
    print('saved to: ' + output_file)
    return output_file

for root, dirs, filenames in os.walk('./data/'):
    for filename in filenames:
        if filename.endswith('identified_nucleotide.fa'):
            nucleotide_input_file_list.append(os.path.join(root, filename))

# viral_data
for nucleotide_input_file in nucleotide_input_file_list:
    if nucleotide_input_file.startswith('./data/viral'):
            
        nucleotide_output_dir = nucleotide_input_file.replace('identified_nucleotide.fa', 'identified_nucleotide/')
        split_fasta_file(nucleotide_input_file, nucleotide_output_dir, sequences_per_file)
        viral_nucleotide_files = [
            os.path.join(nucleotide_output_dir, f)
            for f in os.listdir(nucleotide_output_dir)
            if os.path.isfile(os.path.join(nucleotide_output_dir, f)) 
            and f.endswith('.fa')
            and sum(1 for _ in SeqIO.parse(os.path.join(nucleotide_output_dir, f), "fasta")) == sequences_per_file
        ]
        print(f"{int(sequences_per_file * len(viral_nucleotide_files))} chunked viral sequences were selected")

        with multiprocessing.Pool(processes=8) as pool:
            results = pool.map(process_file_seq, viral_nucleotide_files)

        protein_input_file = nucleotide_input_file.replace('identified_nucleotide.fa', 'identified_protein.fa')
        protein_output_dir = nucleotide_output_dir.replace('identified_nucleotide/', 'identified_protein/')
        split_fasta_file(protein_input_file, protein_output_dir, sequences_per_file)
        viral_protein_files = [
            os.path.join(protein_output_dir, f)
            for f in os.listdir(protein_output_dir)
            if os.path.isfile(os.path.join(protein_output_dir, f))
            and f.endswith('.fa')
            and sum(1 for _ in SeqIO.parse(os.path.join(protein_output_dir, f), "fasta")) == sequences_per_file
        ]
        
        with multiprocessing.Pool(processes=2) as pool:
            results = pool.map(process_file_pro, viral_protein_files)

        ESM_folder = protein_output_dir
        DNABERT_S_folder = nucleotide_output_dir

        files = [f for f in os.listdir(DNABERT_S_folder) if os.path.isfile(os.path.join(DNABERT_S_folder, f)) and f.endswith('.pt')]

        for file in files:
            DNABERT_S_infile = DNABERT_S_folder + file
            ESM_infile = ESM_folder + file.split('_')[0] + '_' + file.split('_')[1] + '_ESM.pt'
            outfile = './data/data_merge/viral.1.1_merged/' + file.split('_')[0] + '_' + file.split('_')[1] + '_merged.pt'
            output_folder = './data/data_merge/viral.1.1_merged/'
            os.makedirs(output_folder, exist_ok=True)
            if not os.path.exists(outfile):
                merge_data(DNABERT_S_infile, ESM_infile, outfile, 'viral')

other_file_list = [item for item in nucleotide_input_file_list if item.endswith('.fa') and not item.startswith('./data/viral')]

random.seed(42)
random_selected_other_nucleotide_files = []
vertebrate_file_list = []
protozoa_file_list = []
plant_file_list = []
invertebrate_file_list = []
fungi_file_list = []
bacteria_file_list = []
archaea_file_list = []

for nucleotide_input_file in other_file_list:
    if nucleotide_input_file.startswith('./data/vertebrate'):
        nucleotide_output_dir = nucleotide_input_file.replace('identified_nucleotide.fa', 'identified_nucleotide/')
        protein_input_file = nucleotide_input_file.replace('identified_nucleotide.fa', 'identified_protein.fa')
        protein_output_dir = protein_input_file.replace('identified_protein.fa', 'identified_protein/')
        split_fasta_file(protein_input_file, protein_output_dir, sequences_per_file)
        split_fasta_file(nucleotide_input_file, nucleotide_output_dir, sequences_per_file)
        vertebrate_file_list.extend([
            os.path.join(nucleotide_output_dir, f) 
            for f in os.listdir(nucleotide_output_dir) 
            if os.path.isfile(os.path.join(nucleotide_output_dir, f)) 
            and f.endswith('.fa')
            and sum(1 for _ in SeqIO.parse(os.path.join(nucleotide_output_dir, f), "fasta")) == sequences_per_file
        ])

    elif nucleotide_input_file.startswith('./data/protozoa'):
        nucleotide_output_dir = nucleotide_input_file.replace('identified_nucleotide.fa', 'identified_nucleotide/')
        protein_input_file = nucleotide_input_file.replace('identified_nucleotide.fa', 'identified_protein.fa')
        protein_output_dir = protein_input_file.replace('identified_protein.fa', 'identified_protein/')
        split_fasta_file(protein_input_file, protein_output_dir, sequences_per_file)
        split_fasta_file(nucleotide_input_file, nucleotide_output_dir, sequences_per_file)
        
        protozoa_file_list.extend([
            os.path.join(nucleotide_output_dir, f) 
            for f in os.listdir(nucleotide_output_dir) 
            if os.path.isfile(os.path.join(nucleotide_output_dir, f))
            and f.endswith('.fa')
            and sum(1 for _ in SeqIO.parse(os.path.join(nucleotide_output_dir, f), "fasta")) == sequences_per_file
            ])
        
    elif nucleotide_input_file.startswith('./data/plant'):
        nucleotide_output_dir = nucleotide_input_file.replace('identified_nucleotide.fa', 'identified_nucleotide/')
        protein_input_file = nucleotide_input_file.replace('identified_nucleotide.fa', 'identified_protein.fa')
        protein_output_dir = protein_input_file.replace('identified_protein.fa', 'identified_protein/')
        split_fasta_file(protein_input_file, protein_output_dir, sequences_per_file)
        split_fasta_file(nucleotide_input_file, nucleotide_output_dir, sequences_per_file)
        
        plant_file_list.extend([
            os.path.join(nucleotide_output_dir, f) 
            for f in os.listdir(nucleotide_output_dir) 
            if os.path.isfile(os.path.join(nucleotide_output_dir, f))
            and f.endswith('.fa')
            and sum(1 for _ in SeqIO.parse(os.path.join(nucleotide_output_dir, f), "fasta")) == sequences_per_file
            ])
        
    elif nucleotide_input_file.startswith('./data/invertebrate'):
        nucleotide_output_dir = nucleotide_input_file.replace('identified_nucleotide.fa', 'identified_nucleotide/')
        protein_input_file = nucleotide_input_file.replace('identified_nucleotide.fa', 'identified_protein.fa')
        protein_output_dir = protein_input_file.replace('identified_protein.fa', 'identified_protein/')
        split_fasta_file(protein_input_file, protein_output_dir, sequences_per_file)
        split_fasta_file(nucleotide_input_file, nucleotide_output_dir, sequences_per_file)
        
        invertebrate_file_list.extend([
            os.path.join(nucleotide_output_dir, f) 
            for f in os.listdir(nucleotide_output_dir) 
            if os.path.isfile(os.path.join(nucleotide_output_dir, f))
            and f.endswith('.fa')
            and sum(1 for _ in SeqIO.parse(os.path.join(nucleotide_output_dir, f), "fasta")) == sequences_per_file
            ])
        
        
    elif nucleotide_input_file.startswith('./data/fungi'):
        nucleotide_output_dir = nucleotide_input_file.replace('identified_nucleotide.fa', 'identified_nucleotide/')
        protein_input_file = nucleotide_input_file.replace('identified_nucleotide.fa', 'identified_protein.fa')
        protein_output_dir = protein_input_file.replace('identified_protein.fa', 'identified_protein/')
        split_fasta_file(protein_input_file, protein_output_dir, sequences_per_file)
        split_fasta_file(nucleotide_input_file, nucleotide_output_dir, sequences_per_file)
        
        fungi_file_list.extend([
            os.path.join(nucleotide_output_dir, f) 
            for f in os.listdir(nucleotide_output_dir) 
            if os.path.isfile(os.path.join(nucleotide_output_dir, f))
            and f.endswith('.fa')
            and sum(1 for _ in SeqIO.parse(os.path.join(nucleotide_output_dir, f), "fasta")) == sequences_per_file
            ])
        
    elif nucleotide_input_file.startswith('./data/bacteria'):
        nucleotide_output_dir = nucleotide_input_file.replace('identified_nucleotide.fa', 'identified_nucleotide/')
        protein_input_file = nucleotide_input_file.replace('identified_nucleotide.fa', 'identified_protein.fa')
        protein_output_dir = protein_input_file.replace('identified_protein.fa', 'identified_protein/')
        split_fasta_file(protein_input_file, protein_output_dir, sequences_per_file)
        split_fasta_file(nucleotide_input_file, nucleotide_output_dir, sequences_per_file)

        bacteria_file_list.extend([
            os.path.join(nucleotide_output_dir, f) 
            for f in os.listdir(nucleotide_output_dir) 
            if os.path.isfile(os.path.join(nucleotide_output_dir, f))
            and f.endswith('.fa')
            and sum(1 for _ in SeqIO.parse(os.path.join(nucleotide_output_dir, f), "fasta")) == sequences_per_file
            ])
        
    elif nucleotide_input_file.startswith('./data/archaea'):
        nucleotide_output_dir = nucleotide_input_file.replace('identified_nucleotide.fa', 'identified_nucleotide/')
        protein_input_file = nucleotide_input_file.replace('identified_nucleotide.fa', 'identified_protein.fa')
        protein_output_dir = protein_input_file.replace('identified_protein.fa', 'identified_protein/')
        split_fasta_file(protein_input_file, protein_output_dir, sequences_per_file)
        split_fasta_file(nucleotide_input_file, nucleotide_output_dir, sequences_per_file)

        archaea_file_list.extend([
            os.path.join(nucleotide_output_dir, f) 
            for f in os.listdir(nucleotide_output_dir) 
            if os.path.isfile(os.path.join(nucleotide_output_dir, f))
            and f.endswith('.fa')
            and sum(1 for _ in SeqIO.parse(os.path.join(nucleotide_output_dir, f), "fasta")) == sequences_per_file
            ])

random_selected_other_nucleotide_files.extend(random.sample(vertebrate_file_list, math.ceil(len(viral_nucleotide_files)/7)))
random_selected_other_nucleotide_files.extend(random.sample(protozoa_file_list, math.ceil(len(viral_nucleotide_files)/7)))
random_selected_other_nucleotide_files.extend(random.sample(plant_file_list, math.ceil(len(viral_nucleotide_files)/7)))
random_selected_other_nucleotide_files.extend(random.sample(invertebrate_file_list, math.ceil(len(viral_nucleotide_files)/7)))
random_selected_other_nucleotide_files.extend(random.sample(fungi_file_list, math.ceil(len(viral_nucleotide_files)/7)))
random_selected_other_nucleotide_files.extend(random.sample(bacteria_file_list, math.ceil(len(viral_nucleotide_files)/7)))
random_selected_other_nucleotide_files.extend(random.sample(archaea_file_list, math.ceil(len(viral_nucleotide_files)/7)))

random_selected_other_nucleotide_files = random.sample(random_selected_other_nucleotide_files, len(viral_nucleotide_files))

random_selected_other_nucleotide_files_feature = []
print(random_selected_other_nucleotide_files)
with multiprocessing.Pool(processes=8) as pool:
    results = pool.map(process_file_seq, random_selected_other_nucleotide_files)
random_selected_other_nucleotide_files_feature.extend(results)

random_selected_other_protein_files = [item.replace('nucleotide', 'protein') for item in random_selected_other_nucleotide_files]
random_selected_other_protein_files_feature = []
print(random_selected_other_protein_files)


with multiprocessing.Pool(processes=2) as pool:
    results = pool.map(process_file_pro, random_selected_other_protein_files)
random_selected_other_protein_files_feature.extend(results)

merged_list = [
    item.replace('./data/', './data/data_merge/')
        .replace('.identified_nucleotide', '_merged')
        .replace('.fa', '_merged.pt')
    for item in random_selected_other_nucleotide_files
]

for DNABERT_S_infile, ESM_infile, merged_file in zip(random_selected_other_nucleotide_files_feature, random_selected_other_protein_files_feature, merged_list):
    print(f'merge {DNABERT_S_infile}, {ESM_infile}')
    output_folder = merged_file.split('output')[0]
    os.makedirs(output_folder, exist_ok=True)
    if not os.path.exists(merged_file):
        merge_data(DNABERT_S_infile, ESM_infile, merged_file, 'host')
