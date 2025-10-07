import os, json, math, argparse, pandas as pd, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import MultiLabelBinarizer
from torchmetrics.functional import f1_score
import esm.inverse_folding
from biotite.structure.io.pdb import PDBFile
from biotite.structure import annotate_sse, sasa, apply_residue_wise
import warnings, tqdm, itertools, tempfile, requests, subprocess

COMPARTMENTS = ['Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane',
                'Mitochondrion', 'Plastid', 'Endoplasmic reticulum',
                'Lysosome/Vacuole', 'Golgi apparatus', 'Peroxisome']
N_COMP = len(COMPARTMENTS)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class LocDataset(Dataset):
    def __init__(self, tsv_path, cdr=0.3):
        df = pd.read_csv(tsv_path, sep='\t')
        self.seqs = df['Sequence'].values
        locs = [s.split('|') for s in df['Locations']]
        self.mlb = MultiLabelBinarizer(classes=COMPARTMENTS)
        self.labels = torch.tensor(self.mlb.fit_transform(locs), dtype=torch.float32)
        self.cdr = cdr  # CD-HIT
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]

def run_esmfold(sequence: str) -> dict:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as fa:
        fa.write(f'>tmp\n{sequence}\n')
        fa_path = fa.name
    pdb_str = subprocess.check_output(['python', '-m', 'esm.inverse_folding.gvp.esmfold',
                                       fa_path], text=True)
    os.unlink(fa_path)
    pdb = PDBFile.read(pdb_str)
    struct = pdb.get_structure()[0]
    # SS
    ss = annotate_sse(struct)  # 'H'/'E'/'C'
    ss3 = [{'H':0,'E':1,'C':2}[s] for s in ss]
    # RSA
    sasa_arr = sasa(struct)
    rsa = sasa_arr / sasa(struct, reference='Wilke')
    # pLDDT
    plddt = [atom.b_factor for atom in struct]
    return dict(plddt=np.array(plddt), ss3=np.array(ss3), rsa=np.array(rsa))

class StructBiasPooler(nn.Module):
    def __init__(self, d_model=640, d_struct=64, window=31, k=128, tau=5):
        super().__init__()
        self.window = window
        self.k, self.tau = k, tau
        self.mlp_struct = nn.Sequential(
            nn.Linear(3+3+1, 128), nn.ReLU(),  # one-hot SS3 + RSA + pLDDT
            nn.Linear(128, d_struct))
        self.proj_q = nn.Linear(d_model, d_struct)
        self.proj_k = nn.Linear(d_model, d_struct)
        self.proj_v = nn.Linear(d_model, d_struct)
        self.lambda_, self.beta = nn.Parameter(torch.tensor(0.05)), nn.Parameter(torch.tensor(0.05))
        self.compress_k = nn.Linear(d_struct, k)
        self.compress_v = nn.Linear(d_struct, k)
        self.head = nn.Linear(d_struct, N_COMP)
    def forward(self, esm_out, struct):
        B, L, _ = esm_out.shape
        plddt, ss3, rsa = struct['plddt'], struct['ss3'], struct['rsa']
        ss_onehot = torch.eye(3, device=DEVICE)[ss3]  # (B,L,3)
        g = torch.cat([ss_onehot, rsa.unsqueeze(-1), plddt.unsqueeze(-1)], -1)  # (B,L,7)
        g = self.mlp_struct(g)  # (B,L,d_struct)
        q, k, v = self.proj_q(esm_out), self.proj_k(esm_out), self.proj_v(esm_out)  # (B,L,d_struct)
        logits = torch.einsum('bld,bmd->blm', q, k) / math.sqrt(g.size(-1))
        bias = self.lambda_ * torch.einsum('bld,bmd->blm', g, g) - \
               self.beta * torch.cdist(g, g, p=2)  # (B,L,L)
        mask = torch.ones(L, L, device=DEVICE).tril(self.window//2).triu(-self.window//2).logical_not()
        logits.masked_fill_(mask, -1e9)
        logits = logits + bias
        k_, v_ = self.compress_k(k), self.compress_v(v)  # (B,L,k)
        attn = torch.softmax(logits, -1)  # (B,L,L)
        context = torch.einsum('blm,bmk->blk', attn, v_)  # (B,L,k)
        w = torch.softmax(plddt / self.tau, -1)  # (B,L)
        z = torch.einsum('bl,blk->bk', w, context)  # (B,k)
        z = torch.tanh(z)
        return torch.sigmoid(self.head(z))  # (B,N_COMP)

class SLAttnESM(nn.Module):
    def __init__(self):
        super().__init__()
        self.esm, self.alphabet = AutoModel.from_pretrained('facebook/esm2-t30-150M-UR50D'), \
                                  AutoTokenizer.from_pretrained('facebook/esm2-t30-150M-UR50D')
        self.pooler = StructBiasPooler()
    def forward(self, tokens, struct):
        with torch.no_grad():
            esm_out = self.esm(tokens).last_hidden_state  # (B,L,640)
        return self.pooler(esm_out, struct)
def weighted_bce_mcc_loss(y_pred, y_true, gamma=0.15):
    w_pos = (y_true==0).sum() / (y_true.sum() + 1e-8)
    w_neg = 1
    bce = nn.functional.binary_cross_entropy(y_pred, y_true,
                                             pos_weight=w_pos, reduction='mean')
    # MCC
    y_pred_bin = (y_pred>0.5).float()
    mcc = []
    for j in range(N_COMP):
        mcc.append(matthews_corrcoef(y_true[:,j].cpu().numpy(),
                                     y_pred_bin[:,j].cpu().numpy()))
    mcc = torch.tensor(mcc, device=y_pred.device).nanmean()
    return bce - gamma * mcc, bce, mcc

def train(args):
    ds = LocDataset(args.tsv)
    train_len = int(0.9*len(ds))
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_len, len(ds)-train_len])
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False)
    model = SLAttnESM().to(DEVICE)
    opt = torch.optim.AdamW(model.pooler.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = len(train_dl)*args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1*steps), steps)
    best = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm.tqdm(train_dl, desc=f'E{epoch}')
        for seqs, labels in pbar:
            labels = labels.to(DEVICE)
            struct = defaultdict(list)
            for s in seqs:
                d = run_esmfold(s)
                for k in d:
                    struct[k].append(torch.tensor(d[k], dtype=torch.float32, device=DEVICE))
            for k in struct:
                struct[k] = torch.stack(struct[k])
            # 2. tokenize
            tokens = model.alphabet(seqs, return_tensors='pt', padding=True)['input_ids'].to(DEVICE)
            # 3. forward
            opt.zero_grad()
            logits = model(tokens, struct)
            loss, bce, mcc = weighted_bce_mcc_loss(logits, labels)
            loss.backward()
            opt.step(); sched.step()
            pbar.set_postfix(loss=loss.item(), bce=bce.item(), mcc=mcc.item())
        # val
        model.eval()
        yp, yt = [], []
        with torch.no_grad():
            for seqs, labels in val_dl:
                labels = labels.to(DEVICE)
                struct = defaultdict(list)
                for s in seqs:
                    d = run_esmfold(s)
                    for k in d:
                        struct[k].append(torch.tensor(d[k], dtype=torch.float32, device=DEVICE))
                for k in struct:
                    struct[k] = torch.stack(struct[k])
                tokens = model.alphabet(seqs, return_tensors='pt', padding=True)['input_ids'].to(DEVICE)
                logits = model(tokens, struct)
                yp.append(logits.cpu()); yt.append(labels.cpu())
        yp, yt = torch.cat(yp), torch.cat(yt)
        micro = f1_score(yp, yt, task='multilabel', num_labels=N_COMP, average='micro')
        macro = f1_score(yp, yt, task='multilabel', num_labels=N_COMP, average='macro')
        print(f'Val  micro-F1={micro:.3f}  macro-F1={macro:.3f}')
        if macro>best:
            best = macro
            torch.save(model.state_dict(), 'sl_attn_esm_best.pt')
    print('Done! best macro-F1=', best)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv', default='uniprotkb_2025_02.tsv')
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    train(args)