from units import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import sys
import pandas as pd

def process_record(record):
    sequence = str(record.seq).upper()
    seqid = record.id
    result = identify_seq(seqid, sequence)
    return result

def determine_virus(group):
    max_score1 = group["score1"].max()
    max_score2 = group["score2"].max()
    return pd.Series({"Is_Virus": max_score2 >= max_score1})  # 直接返回 True/False

class PredictDataBatchDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.ids = []
        self.data = []
        self._load_all_data()

    def _load_all_data(self):
        for file_path in self.file_list:
            data_dict = torch.load(file_path)
            data = data_dict['data']
            self.data.append(data)
            ids = data_dict['ids']
            self.ids.extend(ids)

    def __len__(self):
        return sum(d.size(0) for d in self.data)

    def __getitem__(self, idx):
        cumulative_size = 0
        for data in self.data:
            if cumulative_size + data.size(0) > idx:
                index_in_file = idx - cumulative_size
                return data[index_in_file], self.ids[cumulative_size + index_in_file]
            cumulative_size += data.size(0)
        raise IndexError("Index out of range")
    

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(MLPClassifier, self).__init__()

        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, num_class)

        self.dropout = nn.Dropout(0.5)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x



def predict(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_seqids = []

    with torch.no_grad():
        for batch_data, batch_seqids in data_loader:
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities.data, 1)

            label_map = {1: 'virus', 0: 'others'}
            predicted_labels = [label_map[pred.item()] for pred in predicted]
            
            all_predictions.extend(predicted_labels)
            all_probabilities.extend(probabilities.cpu().numpy())
            all_seqids.extend(batch_seqids)

    return all_seqids, all_predictions, all_probabilities


def make_predictdata(predict_fasta_file, expect_length, model_path):
    sequences_per_file = 10000
    nucleotide_chunked_output = predict_fasta_file.split('.fa')[0] + f'_chunked{expect_length}.fa'

    nucleotide_fasta_file = predict_fasta_file.split('.fa')[0] + '_identified_nucleotide.fa'
    protein_fasta_file = predict_fasta_file.split('.fa')[0] + '_identified_protein.faa'
    nucleotide_output_floder = predict_fasta_file.split('.fa')[0] + '_nucleotide/'
    protein_output_floder = predict_fasta_file.split('.fa')[0] + '_protein/'
    os.makedirs(nucleotide_output_floder, exist_ok=True)
    os.makedirs(protein_output_floder, exist_ok=True)

    fna_list = []
    split_fasta_chunk(predict_fasta_file, nucleotide_chunked_output, int(expect_length))
    
    with open(nucleotide_fasta_file, 'w') as dna_out, open(protein_fasta_file, 'w') as protein_out:
        records = list(SeqIO.parse(nucleotide_chunked_output, 'fasta'))
        for record in tqdm(records, total=len(records)):
            result = process_record(record)
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

    split_fasta_file(nucleotide_fasta_file, nucleotide_output_floder, sequences_per_file)
    nucleotide_file_list_feature = []
    nucleotide_file_list = [
        os.path.join(nucleotide_output_floder, f) 
        for f in os.listdir(nucleotide_output_floder) 
        if os.path.isfile(os.path.join(nucleotide_output_floder, f)) and f.endswith('.fa')
        ]
    for file in nucleotide_file_list:
        extract_DNABERT_S(input_file=file, out_file=f'{file.split(".fa")[0]}_DNABERT_S.pt')
        print ('saved to: ' + f'{file.split(".fa")[0]}_DNABERT_S.pt')
        nucleotide_file_list_feature.append(f'{file.split(".fa")[0]}_DNABERT_S.pt')

    split_fasta_file(protein_fasta_file, protein_output_floder, sequences_per_file)
    protein_feature = []
    protein_file_list = [
            os.path.join(protein_output_floder, f) 
            for f in os.listdir(protein_output_floder) 
            if os.path.isfile(os.path.join(protein_output_floder, f)) and f.endswith('.fa')
            ]
    for file in protein_file_list:
        extract_esm(fasta_file = file, out_file = f'{file.split(".fa")[0]}_ESM.pt')
        print ('saved to: ' + f'{file.split(".fa")[0]}_ESM.pt')
        protein_feature.append(f'{file.split(".fa")[0]}_ESM.pt')
    
    merged_list = [
        item.replace('nucleotide', 'merged')
            .replace('.fa', '_merged.pt') 
        for item in nucleotide_file_list
    ]


    for DNABERT_S_infile, ESM_infile, merged_file in zip(nucleotide_file_list_feature, protein_feature, merged_list):
        print(DNABERT_S_infile, ESM_infile)
        output_folder = merged_file.split('output_')[0]
        os.makedirs(output_folder, exist_ok=True)
        merge_data(DNABERT_S_infile, ESM_infile, merged_file)

    predict_dataset = PredictDataBatchDataset(merged_list)
    predict_loader = DataLoader(predict_dataset, batch_size=256, shuffle=False, num_workers=4)
    mlp_model = torch.load(model_path, weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_model = mlp_model.to(device)
    seqids, predictions, probabilities = predict(mlp_model, predict_loader, device)

    output_file = output_folder + 'prediction_results.txt'

    with open(output_file, 'w') as f:
        f.write("Sequence_ID\tPrediction\tscore1\tscore2\n")
        for seqid, prediction, probability in zip(seqids, predictions, probabilities):
            prob_str = '\t'.join(map(str, probability))
            f.write(f"{seqid}\t{prediction}\t{prob_str}\n")

    print(f"Predictions saved to {output_file}")

    df = pd.read_csv(output_file, sep="\t")

    df["Modified_ID"] = df["Sequence_ID"].str[:-2]

    df = df.groupby("Modified_ID", group_keys=False).apply(determine_virus).reset_index()

    df.to_csv(output_folder + "prediction_results_highestscore.csv", sep="\t", index=False)

    print("Prediction results with highest sore saved to 'prediction_results_highestscore.csv'")

predict_fasta_file = sys.argv[1]
expect_length = sys.argv[2]
model_path = sys.argv[3]
if __name__ == "__main__":
    make_predictdata(predict_fasta_file, expect_length, model_path)


