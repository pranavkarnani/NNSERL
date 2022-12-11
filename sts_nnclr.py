import os
import glob
import math
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import csv
import gzip
import random

from sentence_transformers import InputExample
from transformers import BertModel, BertTokenizer
# from torchsummaryX import summary
import random
from tqdm import tqdm
import wandb
import json
from typing import Optional

from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import pearsonr, spearmanr

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    'model_name': 'bert-base-uncased',
    'augmentation_type_1': 'shuffle',
    'augmentation_type_2': 'cutoff',
    'BATCH_SIZE': 96,
    'models': os.getcwd() + '/model',
    'data_path': './downstream',
    'cutoff_direction': 'column',
    'cutoff_rate': 0.2,
    'lr': 5e-7,
    'weight_decay': 0,
    'temperature': 0.1,
    'LARGE_NUM': 1e9,
    'hidden_norm': True,
    'total_epochs': 10,
    'max_length': 64,
    'num_warmup_steps': 2000,
    'num_training_steps': 13000,
    'num_labels': 3,
    'mixup_rate': 0.4
}

nli_dataset_path = 'datasets/AllNLI.tsv.gz'
sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'
if not os.path.exists(nli_dataset_path):
    util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)
if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

try:
    os.mkdir(os.getcwd() + '/model')
except:
    pass

def load_sts12(need_label = False, use_all_unsupervised_texts=True, no_pair=True):
    dataset_names = ["MSRpar", "MSRvid", "SMTeuroparl", "surprise.OnWN", "surprise.SMTnews"]
    return load_sts(need_label, "12", dataset_names, no_pair=no_pair)
    
def load_sts13(need_label = False, use_all_unsupervised_texts=True, no_pair=True):
    dataset_names = ["headlines", "OnWN", "FNWN"]
    return load_sts(need_label, "13", dataset_names, no_pair=no_pair)

def load_sts14(need_label = False, use_all_unsupervised_texts=True, no_pair=True):
    dataset_names = ["images", "OnWN", "tweet-news", "deft-news", "deft-forum", "headlines"]
    return load_sts(need_label, "14", dataset_names, no_pair=no_pair)

def load_sts15(need_label = False, use_all_unsupervised_texts=True, no_pair=True):
    dataset_names = ["answers-forums", "answers-students", "belief", "headlines", "images"]
    return load_sts(need_label, "15", dataset_names, no_pair=no_pair)

def load_sts16(need_label = False, use_all_unsupervised_texts=True, no_pair=True):
    dataset_names = ["answer-answer", "headlines", "plagiarism", "postediting", "question-question"]
    return load_sts(need_label, "16", dataset_names, no_pair=no_pair)

def load_sts(need_label, year, dataset_names, no_pair=False):
    
    all_samples = []
    sts_data_path = f"{config['data_path']}/STS/STS{year}-en-test"
    
    for dataset_name in dataset_names:
        input_file = os.path.join(sts_data_path, f"STS.input.{dataset_name}.txt")
        label_file = os.path.join(sts_data_path, f"STS.gs.{dataset_name}.txt")
        sub_samples = load_paired_samples(need_label, input_file, label_file, no_pair=no_pair)
        all_samples.extend(sub_samples)
    
    return all_samples

def load_paired_samples(need_label, input_file, label_file, scale=5.0, no_pair=False):

    samples = []

    with open(input_file, "r") as f:
        input_lines = f.readlines()

    label_lines = [None]*len(input_lines)
    if label_file:
        with open(label_file, 'r') as labels:
            label_lines = labels.readlines()

    if need_label:
        new_input_lines, new_label_lines = [], []
        for idx in range(len(label_lines)):
            label_lines[idx] = label_lines[idx].strip()
            if label_lines[idx]:
                new_input_lines.append(input_lines[idx])
                new_label_lines.append(label_lines[idx].strip())
        input_lines = new_input_lines
        label_lines = new_label_lines

    for input_line, label_line in zip(input_lines, label_lines):
        sentences = input_line.split("\t")
            
        if len(sentences)==2:
            sent1, sent2 = sentences
        else:
            sent1, sent2 = sentences[0], None

        if need_label:
            samples.append(InputExample(texts=[sent1, sent2], label=float(label_line)/scale))

        else:
            if no_pair:
                samples.append(InputExample(texts=[sent1]))
                if sent2:
                    samples.append(InputExample(texts=[sent2]))
            else:
                samples.append(InputExample(texts=[sent1, sent2]))
    return samples

def load_stsbenchmark(need_label=False, use_all_unsupervised_texts=True, no_pair=True):

    all_samples = []
    if use_all_unsupervised_texts:
        splits = ["train", "dev", "test"]
    else:
        splits = ["test"]
    
    for split in splits:
        sts_benchmark_data_path = f"{config['data_path']}/STS/STSBenchmark/sts-{split}.csv"
        
        samples = []
        with open(sts_benchmark_data_path, "r") as f:
            lines = f.readlines()
        
            for line in lines:
                line = line.strip()
                _, _, _, _, label, sent1, sent2 = line.split("\t")
                if need_label:
                    samples.append(InputExample(texts=[sent1, sent2], label=float(label) / 5.0))
                else:
                    if no_pair:
                        samples.append(InputExample(texts=[sent1]))
                        samples.append(InputExample(texts=[sent2]))
                    else:
                        samples.append(InputExample(texts=[sent1, sent2]))
        all_samples.extend(samples)
    
    return all_samples

def load_sickr(need_label=False, use_all_unsupervised_texts=True, no_pair=True):
    
    all_samples = []
    if use_all_unsupervised_texts:
        splits = ["train", "trial", "test_annotated"]
    else:
        splits = ["test_annotated"]

    for split in splits:
        samples = []
        sick_data_path = f"{config['data_path']}/SICK/SICK_{split}.txt"
        
        with open(sick_data_path, "r") as f:
            lines = f.readlines()
        
        for line in lines[1:]:
            line = line.strip()
            _, sent1, sent2, label, _ = line.split("\t")
            
            if need_label:
                samples.append(InputExample(texts=[sent1, sent2], label=float(label) / 5.0))
            else:
                if no_pair:
                    samples.append(InputExample(texts=[sent1]))
                    samples.append(InputExample(texts=[sent2]))
                else:
                    samples.append(InputExample(texts=[sent1, sent2]))
        all_samples.extend(samples)
    
    return all_samples

def eval_sts(year, dataset_names):
    sts_data_path = f"./downstream/STS/STS{year}-en-test"
    
    all_samples = []
    for dataset_name in dataset_names:
        input_file = os.path.join(sts_data_path, f"STS.input.{dataset_name}.txt")
        label_file = os.path.join(sts_data_path, f"STS.gs.{dataset_name}.txt")
        sub_samples = load_paired_samples(True, input_file, label_file)
        all_samples.extend(sub_samples)
    
    return all_samples

def eval_sts12():
    dataset_names = ["MSRpar", "MSRvid", "SMTeuroparl", "surprise.OnWN", "surprise.SMTnews"]
    return eval_sts("12", dataset_names)
    
def eval_sts13():
    dataset_names = ["headlines", "OnWN", "FNWN"]
    return eval_sts("13", dataset_names)

def eval_sts14():
    dataset_names = ["images", "OnWN", "tweet-news", "deft-news", "deft-forum", "headlines"]
    return eval_sts("14", dataset_names)

def eval_sts15():
    dataset_names = ["answers-forums", "answers-students", "belief", "headlines", "images"]
    return eval_sts("15", dataset_names)

def eval_sts16():
    dataset_names = ["answer-answer", "headlines", "plagiarism", "postediting", "question-question"]
    return eval_sts("16", dataset_names)

def eval_stsbenchmark():
    print("Evaluation on STSBenchmark dataset")
    sts_benchmark_data_path = "./downstream/STS/STSBenchmark/sts-test.csv"
    with open(sts_benchmark_data_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    samples = []
    for line in lines:
        _, _, _, _, label, sent1, sent2 = line.split("\t")
        samples.append(InputExample(texts=[sent1, sent2], label=float(label) / 5.0))
    print(f"Loaded examples from STSBenchmark dataset, total {len(samples)} examples")
    return samples

def eval_sickr():
    print("Evaluation on SICK (relatedness) dataset")
    sick_data_path = "./downstream/SICK/SICK_test_annotated.txt"
    with open(sick_data_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    samples = []
    for line in lines[1:]:
        _, sent1, sent2, label, _ = line.split("\t")
        samples.append(InputExample(texts=[sent1, sent2], label=float(label) / 5.0))
    print(f"Loaded examples from SICK dataset, total {len(samples)} examples")
    
    return samples

class STSDatasetUnsupervised(Dataset):
    def __init__(self):
        super(STSDatasetUnsupervised, self).__init__()

        sts_data = []
        sts_data_12 = load_sts12()
        sts_data_13 = load_sts13()
        sts_data_14 = load_sts14()
        sts_data_15 = load_sts15()
        sts_data_16 = load_sts16()
        stsb = load_stsbenchmark()
        sickr = load_sickr()
        
        sts_data.extend(sts_data_12)
        sts_data.extend(sts_data_13)
        sts_data.extend(sts_data_14)
        sts_data.extend(sts_data_15)
        sts_data.extend(sts_data_16)
        sts_data.extend(stsb)
        sts_data.extend(sickr)

        self.dataset = sts_data
        self.length = len(self.dataset)

    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        return self.dataset[index]

    def collate(batch):
        num_texts = len(batch[0].texts)
        texts = []

        for example in batch:
            texts.append(example.texts[0])
        return tokenizer.batch_encode_plus(texts, padding='max_length', return_tensors='pt', truncation=True)

class STSDatasetUnsupervisedVal(Dataset):
    def __init__(self):
        super(STSDatasetUnsupervisedVal, self).__init__()
        self.dataset = []
        sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'
        with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if row['split'] == 'dev':
                    score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
                    self.dataset.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        
        self.length = len(self.dataset)

    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        return self.dataset[index]

    def collate(batch):
        num_texts = len(batch[0].texts)
        texts1, texts2 = [], []
        labels = []

        for example in batch:
            texts1.append(example.texts[0])
            texts2.append(example.texts[1])
            labels.append(example.label)

        return (tokenizer.batch_encode_plus(texts1, padding='max_length', return_tensors='pt', truncation=True),
        tokenizer.batch_encode_plus(texts2, padding='max_length', return_tensors='pt', truncation=True),
        labels)

class STSDatasetUnsupervisedTest(Dataset):
    def __init__(self):
        super(STSDatasetUnsupervisedTest, self).__init__()

        sts_data = []
        sts_data_12 = eval_sts12()
        sts_data_13 = eval_sts13()
        sts_data_14 = eval_sts14()
        sts_data_15 = eval_sts15()
        sts_data_16 = eval_sts16()
        stsb = eval_stsbenchmark()
        sickr = eval_sickr()
        
        sts_data.extend(sts_data_12)
        sts_data.extend(sts_data_13)
        sts_data.extend(sts_data_14)
        sts_data.extend(sts_data_15)
        sts_data.extend(sts_data_16)
        sts_data.extend(stsb)
        sts_data.extend(sickr)

        self.dataset = sts_data
        self.length = len(self.dataset)

    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        return self.dataset[index]

    def collate(batch):
        num_texts = len(batch[0].texts)
        texts1, texts2 = [], []
        labels = []

        for example in batch:
            texts1.append(example.texts[0])
            texts2.append(example.texts[1])
            labels.append(example.label)

        return (tokenizer.batch_encode_plus(texts1, padding='max_length', return_tensors='pt', truncation=True),
        tokenizer.batch_encode_plus(texts2, padding='max_length', return_tensors='pt', truncation=True),
        labels)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', model_max_length=config['max_length'])

unsupervised_dataset = STSDatasetUnsupervised()
unsupervised_dataset_val = STSDatasetUnsupervisedVal()
unsupervised_dataset_test = STSDatasetUnsupervisedTest()
train_dataloader = DataLoader(unsupervised_dataset, shuffle = True, batch_size=config['BATCH_SIZE'], collate_fn=STSDatasetUnsupervised.collate)
val_dataloader = DataLoader(unsupervised_dataset_val, shuffle = False, batch_size=config['BATCH_SIZE'], collate_fn=STSDatasetUnsupervisedVal.collate)
test_dataloader = DataLoader(unsupervised_dataset_test, shuffle = False, batch_size=config['BATCH_SIZE'], collate_fn=STSDatasetUnsupervisedTest.collate)

print("# of records in the train unsupervised dataset:", unsupervised_dataset.length)
print("# of records in the val unsupervised dataset:",unsupervised_dataset_val.length)
print("# of records in the test unsupervised dataset:",unsupervised_dataset_test.length)

for i, x in enumerate(train_dataloader):
    break


"""## Loss Functions"""

class NTXENT(nn.Module):
    def __init__(self, temperature, LARGE_NUM, hidden_norm):
        self.temperature = temperature
        self.LARGE_NUM = LARGE_NUM
        self.hidden_norm = hidden_norm
        self.batch_size = config['BATCH_SIZE']

    def __call__(self, hidden1, hidden2):
        if self.hidden_norm:
            hidden1 = torch.nn.functional.normalize(hidden1, p=2, dim=-1)
            hidden2 = torch.nn.functional.normalize(hidden2, p=2, dim=-1)

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = torch.arange(0, hidden1.shape[0]).to(device=hidden1.device)
        masks = torch.nn.functional.one_hot(torch.arange(0, hidden1.shape[0]), num_classes=hidden1.shape[0]).to(device=hidden1.device, dtype=torch.float)

        masks = torch.where(masks == 1, 0.8, 0.2)

        logits_aa = torch.matmul(hidden1, hidden1_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
        logits_aa = logits_aa - masks * self.LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
        logits_bb = logits_bb - masks * self.LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
        logits_ba = torch.matmul(hidden2, hidden1_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)

        loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
        loss = loss_a + loss_b
        return loss


class ArcFace(nn.Module):

    def __init__(self, temperature, m=0.5, mixup = False, mixup_rate = 0.0):
        super(ArcFace, self).__init__()

        self.temperature = temperature
        self.cos_margin = np.cos(m)
        self.sin_margin = np.sin(m)
        self.mixup = mixup
        self.mixup_rate = mixup_rate

    def forward(self, hidden1, hidden2):

        hidden1 = torch.nn.functional.normalize(hidden1, p=2, dim=-1)
        hidden2 = torch.nn.functional.normalize(hidden2, p=2, dim=-1)

        cos_theta = torch.matmul(hidden1, hidden2.T) # bsz x bsz
        mask = torch.eye(hidden1.shape[0]).to(device) # bsz x bsz

        sin_theta = torch.sqrt(1 - torch.pow(cos_theta, 2)) # bsz x bsz
        cos_theta_m = cos_theta * self.cos_margin - sin_theta * self.sin_margin # bsz x bsz

        num = torch.sum(mask * (torch.exp((cos_theta_m / self.temperature))), dim = 1) # bsz
        denom = num + torch.sum((1 - mask) * (torch.exp(cos_theta / self.temperature)), dim = 1)
        loss = (-1 * (torch.log(num / denom))).mean()
        return loss


class Network(nn.Module):
    def __init__(self, input_dim_projection, hidden_dim_projection, output_dim_projection, hidden_dim_prediction, lamb, mixup):
        super(Network, self).__init__()

        augmentation_args = {'direction':config['cutoff_direction'], 'cutoff_rate':config['cutoff_rate']}

        self.encoder = BertModel.from_pretrained('bert-base-uncased', hidden_dropout_prob = 0, attention_probs_dropout_prob = 0, **augmentation_args, output_hidden_states=True)
        self.encoder.model_max_len = config['max_length']
        self.feature_queue = torch.nn.functional.normalize(torch.randn(size=(80000, 768), requires_grad=False), p=2.0, dim = 1).to(device)
        self.lamb = lamb
        self.mixup = mixup

        # self.projection_MLP = nn.Sequential(
        #     nn.Linear(input_dim_projection, hidden_dim_projection, bias=False),
        #     nn.Linear(hidden_dim_projection, output_dim_projection, bias=True),
        #     nn.GELU(),
        #     nn.BatchNorm1d(output_dim_projection)
        # )

    def forward(self, x1, x2=None, mode = 'train', version = None):
        if x2 is None:
            x2 = x1

        input_ids1, attention_mask1, token_type_ids1 = x1['input_ids'].to(device), x1['attention_mask'].to(device), x1['token_type_ids'].to(device)
        input_ids2, attention_mask2, token_type_ids2 = x2['input_ids'].to(device), x2['attention_mask'].to(device), x2['token_type_ids'].to(device)

        embedding_output1 = self.encoder(input_ids = input_ids1, attention_mask = attention_mask1, token_type_ids = token_type_ids1, augmentation_type = 'shuffle')
        embedding_output2 = self.encoder(input_ids = input_ids2, attention_mask = attention_mask2, token_type_ids = token_type_ids2, augmentation_type = 'cutoff')

        if mode == 'test':
            if version == "last2":
                embedding_output1 = (embedding_output1['hidden_states'][-1] + embedding_output1['hidden_states'][-2])/2
                embedding_output2 = (embedding_output2['hidden_states'][-1] + embedding_output2['hidden_states'][-2])/2
            elif version == "first&last":
                embedding_output1 = (embedding_output1['hidden_states'][-1] + embedding_output1['hidden_states'][0])/2
                embedding_output2 = (embedding_output2['hidden_states'][-1] + embedding_output2['hidden_states'][0])/2
        else:
            embedding_output1 = embedding_output1[0]
            embedding_output2 = embedding_output2[0]

        sentence_embedding_1 = self.mean_pooling(embedding_output1, attention_mask1)
        sentence_embedding_2 = self.mean_pooling(embedding_output2, attention_mask2)

        return sentence_embedding_1, sentence_embedding_2

    def update_queue(self, projections):
        projections = torch.nn.functional.normalize(projections, p=2, dim=-1)
        self.feature_queue = torch.concat([projections, self.feature_queue[:-config['BATCH_SIZE']]], axis=0)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def find_nearest_neighbors(self, projections):
        projections = torch.nn.functional.normalize(projections, p=2, dim=-1)
        support_similarities = torch.matmul(projections, self.feature_queue.T)
        indices = torch.argmax(support_similarities, dim=-1)
        return projections + (self.feature_queue[indices] - projections).detach()

model = Network(768, 2048, 2048, 2048, config['mixup_rate'], False).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
criterion_ntxent = NTXENT(config['temperature'], config['LARGE_NUM'], config['hidden_norm'])
criterion_arcface = ArcFace(config['temperature'], 0.2, True)

# wandb.login(key="enter key")
# run = wandb.init(project="", entity="", name = "", reinit = True)

# checkpoint = torch.load('./checkpoint2.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def train(epoch):
    batch_bar = tqdm(total=len(train_dataloader), dynamic_ncols=True, position=0, leave=False, desc='Train')
    total_loss = 0
    for i, x in enumerate(train_dataloader):
        model.train()
        optimizer.zero_grad()
       
        projection1, projection2 = model(x, None)
        projection2 = 0.2 * projection1 + 0.8 * projection2
        if epoch is None:
            loss =  criterion_ntxent(projection1, projection2)
        else:
            neighbor1 = model.find_nearest_neighbors(projection1)
            neighbor2 = model.find_nearest_neighbors(projection2)
            loss = criterion_arcface(neighbor1, projection2) / 2 + criterion_arcface(neighbor2, projection1) / 2

        total_loss += loss.item()
        loss.backward()

        model.update_queue(projection1)
        optimizer.step()

        batch_bar.set_postfix(
            epoch="{:d}".format(epoch),
            loss="{:.04f}".format(loss.item()))
        batch_bar.update()

        if i % 100 == 0 and i != 0:
            epc, esc, epd, esd = val()

            wandb.log({"train_loss": total_loss/(i//100), 
                       "Pearson Cosine:": epc, 
                       "Spearman Cosine": esc, 
                       "Pearson Dot": epd, 
                       "Spearman Dot": esd
            })

            print("Epoch number: ", epoch)
            print("Val Pearson Cosine:", epc)
            print("Val Spearman Cosine:", esc)
            print("Val Pearson Dot:", epd)
            print("Val Spearman Dot:", esd)
            print()
    return total_loss / len(train_dataloader)

def val():
    model.eval()
    batch_bar = tqdm(total=len(val_dataloader), dynamic_ncols=True, position=0, leave=False, desc='Val')
    total_loss = 0
    all_embedding1 = []
    all_embedding2 = []
    all_labels = []
    for i, (x1, x2, labels) in enumerate(val_dataloader):
        with torch.no_grad():
            embedding1, embedding2 = model(x1, x2)

        for emb1 in embedding1.detach().cpu().numpy():
            all_embedding1.append(emb1)
        for emb2 in embedding2.detach().cpu().numpy():
            all_embedding2.append(emb2)
        [all_labels.append(label) for label in labels]
        batch_bar.set_postfix(
                iterations="{:2f}".format(i)
        )
        batch_bar.update()

    cosine_scores = [1-i for i in paired_cosine_distances(all_embedding1, all_embedding2)]
    dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(all_embedding1, all_embedding2)]

    eval_pearson_cosine, _ = pearsonr(np.array(all_labels), cosine_scores)
    eval_spearman_cosine, _ = spearmanr(np.array(all_labels), cosine_scores)

    eval_pearson_dot, _ = pearsonr(np.array(all_labels), dot_products)
    eval_spearman_dot, _ = spearmanr(np.array(all_labels), dot_products)

    return eval_pearson_cosine, eval_spearman_cosine, eval_pearson_dot, eval_spearman_dot

def test(version):
    model.eval()
    all_embedding1 = []
    all_embedding2 = []
    all_labels = []
    batch_bar = tqdm(total=len(test_dataloader), dynamic_ncols=True, position=0, leave=False, desc='Test')
    for i, (x1, x2, labels) in enumerate(test_dataloader):
        with torch.no_grad():
            embedding1, embedding2 = model(x1, x2, mode = 'test', version = version)
        for emb1 in embedding1.detach().cpu().numpy():
            all_embedding1.append(emb1)
        for emb2 in embedding2.detach().cpu().numpy():
            all_embedding2.append(emb2)
        [all_labels.append(label) for label in labels]
        batch_bar.set_postfix(
                iterations="{:2f}".format(i)
        )
        batch_bar.update()

        cosine_scores = [1-i for i in paired_cosine_distances(all_embedding1, all_embedding2)]
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(all_embedding1, all_embedding2)]

        test_pearson_cosine, _ = pearsonr(np.array(all_labels), cosine_scores)
        test_spearman_cosine, _ = spearmanr(np.array(all_labels), cosine_scores)

        test_pearson_dot, _ = pearsonr(np.array(all_labels), dot_products)
        test_spearman_dot, _ = spearmanr(np.array(all_labels), dot_products)

    return test_pearson_cosine, test_spearman_cosine, test_pearson_dot, test_spearman_dot

num_epochs = config['total_epochs']
print('\n\n\n\n\n')
print('Training')

# wandb.log(config)

for i in range(1, num_epochs + 1):
    train_loss = train(i)

    # wandb.log({"train_loss": train_loss})
    # wandb.save("checkpoint" + str(i) + ".pth")

    print("Train loss is ", train_loss)
    epc, esc, epd, esd = test(version = "first&last")
    print("Test Pearson Cosine:", epc)
    print("Test Spearman Cosine:", esc)
    print("Test Pearson Dot:", epd)
    print("Test Spearman Dot:", esd)


    epc, esc, epd, esd = test(version = "last2")
    print("Test Pearson Cosine:", epc)
    print("Test Spearman Cosine:", esc)
    print("Test Pearson Dot:", epd)
    print("Test Spearman Dot:", esd)

    torch.save({'model_state_dict':model.state_dict(),
                  'optimizer_state_dict':optimizer.state_dict(),
                  'epoch': i}, f"./checkpoint{i}.pth")

    # wandb.save(f"checkpoint{i}.pth")

    # wandb.log({
    #     "Test Pearson Cosine:": epc, 
    #     "Test Spearman Cosine": esc, 
    #     "Test Pearson Dot": epd, 
    #     "Test Spearman Dot": esd
    # })

    print()

run.finish()