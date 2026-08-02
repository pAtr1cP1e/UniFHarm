from rdkit.Chem.Draw import rdMolDraw2D
from transformers import AutoTokenizer, EsmModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import dill
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import os


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, query, key, value, key_mask=None):
        batch_size, q_len, dim = query.shape
        k_len = key.size(1)

        residual = query

        Q = self.q_proj(query)  # [B, q_len, dim]
        K = self.k_proj(key)  # [B, k_len, dim]
        V = self.v_proj(value)  # [B, k_len, dim]

        Q = Q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, heads, q_len, k_len]

        if key_mask is not None:
            mask = key_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # [B, heads, q_len, head_dim]

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, dim)
        attn_output = self.out_proj(attn_output)

        output = self.layer_norm(residual + attn_output)

        residual = output
        output = self.ffn_norm(residual + self.ffn(output))

        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.layer_norm(x + self.net(x))


class Expert(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MoELayer(nn.Module):
    def __init__(self, dim, num_experts=3, num_shared=1,
                 top_k=2, dropout=0.1,
                 load_balance_coef=0.001):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_shared = num_shared
        self.num_routed = num_experts - num_shared
        self.top_k = top_k
        self.load_balance_coef = load_balance_coef

        self.shared_experts = nn.ModuleList([
            Expert(dim, dim * 4, dropout) for _ in range(num_shared)
        ])
        self.routed_experts = nn.ModuleList([
            Expert(dim, dim * 4, dropout) for _ in range(self.num_routed)
        ])

        self.gate = nn.Linear(dim, self.num_routed)
        self.layer_norm = nn.LayerNorm(self.dim)

    def forward(self, x):
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, dim = x.shape
            x_flat = x.view(-1, self.dim)
        else:
            batch_size, dim = x.shape
            seq_len = 1
            x_flat = x

        router_logits = self.gate(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        aux_loss = 0.0
        if self.training:
            aux_loss = self._compute_load_balance_loss(router_probs, top_k_indices)

        output = torch.zeros_like(x_flat)

        for shared_expert in self.shared_experts:
            output = output + shared_expert(x_flat)

        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_weight = top_k_probs[:, i:i + 1]

            for exp_id in range(self.num_routed):
                mask = (expert_idx == exp_id).unsqueeze(-1)
                if mask.any():
                    expert_input = x_flat * mask.float()
                    expert_out = self.routed_experts[exp_id](expert_input)
                    output = output + expert_out * expert_weight * mask.float()

        output = self.layer_norm(x_flat + output)

        if len(original_shape) == 3:
            output = output.view(batch_size, seq_len, dim)

        return output, aux_loss

    def _compute_load_balance_loss(self, router_probs, top_k_indices):
        expert_mask = F.one_hot(top_k_indices, self.num_routed).sum(dim=1).float()
        expert_fraction = expert_mask.mean(dim=0)  # [num_routed]
        avg_router_prob = router_probs.mean(dim=0)  # [num_routed]
        balance_loss = self.num_routed * (expert_fraction * avg_router_prob).sum()
        return self.load_balance_coef * balance_loss

class BidirectionalCrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()

        self.mol_cross_attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.prot_cross_attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.mol_ffn = FeedForward(dim, dropout)
        self.prot_ffn = FeedForward(dim, dropout)

    def forward(self, mol_repr, prot_repr, mol_mask, prot_mask):
        new_mol = self.mol_cross_attn(
            query=mol_repr,
            key=prot_repr,
            value=prot_repr,
            key_mask=prot_mask
        )

        mol_repr = mol_repr + new_mol

        new_prot = self.prot_cross_attn(
            query=prot_repr,
            key=mol_repr,
            value=mol_repr,
            key_mask=mol_mask
        )

        prot_repr = prot_repr + new_prot

        mol_repr = mol_repr + self.mol_ffn(mol_repr)
        prot_repr = prot_repr + self.prot_ffn(prot_repr)

        return mol_repr, prot_repr


class UniEncoder(nn.Module):
    def __init__(
            self,
            bermol_path: str = r'bermol/BerMolModel_base.pkl',
            esm_path: str = r'esm2_t33_650M_UR50D',
            max_length: int = 1024,
            projection_dim: int = 640,
            num_attention_heads: int = 8,
            num_cross_layers: int = 4,
            device: str = r'cuda'
    ):
        super().__init__()
        self.max_length = max_length
        self.proj_dim = projection_dim
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(bermol_path, 'rb') as f:
            self.mol_enc = dill.load(f)
        self.mol_enc.model = self.mol_enc.model.to(device)
        self.mol_enc.device = device
        for param in self.mol_enc.model.parameters():
            param.requires_grad = False
        self.mol_enc.model.eval()
        self.prot_tokenizer = AutoTokenizer.from_pretrained(r'esm2_t33_650M_UR50D')
        self.prot_enc = EsmModel.from_pretrained(esm_path).to(device)
        for param in self.prot_enc.parameters():
            param.requires_grad = False
        self.prot_enc.eval()

        self.mol_proj = nn.Sequential(
            nn.Linear(768, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.prot_proj = nn.Sequential(
            nn.Linear(1280, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.mol_pos_embed = nn.Parameter(torch.randn(1, self.max_length, self.proj_dim))
        self.prot_pos_embed = nn.Parameter(torch.randn(1, self.max_length, self.proj_dim))

        self.cross_attn_layers = nn.ModuleList([
            BidirectionalCrossAttentionBlock(self.proj_dim, num_attention_heads)
            for _ in range(num_cross_layers)
        ])

        self.mol_pooler = nn.Sequential(
            nn.Linear(self.proj_dim, self.proj_dim),
            nn.Tanh(),
        )

        self.prot_pooler = nn.Sequential(
            nn.Linear(self.proj_dim, self.proj_dim),
            nn.Tanh(),
        )

    def encode_molecules(self, smiles_list):
        mol_out = []
        mask_out = []
        with torch.no_grad():
            for smiles in smiles_list:
                mol_repr, _ = self.mol_enc.transform(smiles, device=self.device)
                mol_repr = mol_repr.squeeze()[1:-1].to(self.device)
                if mol_repr.shape[0] > self.max_length:
                    seq_out = mol_repr[:self.max_length]
                    mask = torch.ones(self.max_length, device=self.device)
                else:
                    padding = torch.zeros(self.max_length - mol_repr.shape[0], 768, device=self.device)
                    seq_out = torch.cat([mol_repr, padding], dim=0)
                    mask = torch.cat([
                        torch.ones(mol_repr.shape[0], device=self.device),
                        torch.zeros(self.max_length - mol_repr.shape[0], device=self.device)
                    ])
                mol_out.append(seq_out)
                mask_out.append(mask)
        mol_hidden = torch.stack(mol_out).to(self.device)
        mol_mask = torch.stack(mask_out).to(self.device)

        mol_hidden = self.mol_proj(mol_hidden)
        mol_hidden = mol_hidden + self.mol_pos_embed[:, :self.max_length, :]
        return mol_hidden, mol_mask.to(self.device)

    def encode_protein(self, sequence):
        with torch.no_grad():
            tokens = self.prot_tokenizer(
                sequence,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)
        prot_hidden = self.prot_enc(**tokens).last_hidden_state
        prot_hidden = self.prot_proj(prot_hidden)

        prot_hidden = prot_hidden + self.prot_pos_embed[:, :self.max_length, :]
        return prot_hidden, tokens['attention_mask']

    def forward(self, batch):
        mol_hidden, mol_mask = self.encode_molecules(batch['mol'])
        prot_hidden, prot_mask = self.encode_protein(batch['prot'])

        for cross_layer in self.cross_attn_layers:
            mol_repr, prot_repr = cross_layer(
                mol_repr=mol_hidden,
                prot_repr=prot_hidden,
                mol_mask=mol_mask,
                prot_mask=prot_mask
            )

        mol_pool = self.mol_pooler(mol_repr.mean(dim=1))
        prot_pool = self.prot_pooler(prot_repr.mean(dim=1))
        return mol_repr, prot_repr, mol_pool, prot_pool


class DownStreamModel(nn.Module):
    def __init__(self, task='dti', device=None):
        super().__init__()
        self.task = task
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = UniEncoder(device=device, num_cross_layers=4)

        self.moe = MoELayer(self.encoder.proj_dim * 2)

        if self.task in ['dti', 'moa']:
            self.predictor = nn.Sequential(
                nn.Linear(self.encoder.proj_dim * 2, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 2)
            )
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.predictor = nn.Sequential(
                nn.Linear(self.encoder.proj_dim * 2, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 1)
            )
            self.criterion = nn.MSELoss()

    def forward(self, batch):
        batch['label'] = batch['label'].to(self.device)
        mol_feat, prot_feat, _, _ = self.encoder(batch)
        fusion, aux_loss = self.moe(torch.cat((mol_feat, prot_feat), dim=-1))
        out = self.predictor(fusion.mean(dim=1))
        probs, preds = None, None
        loss = self.criterion(out.squeeze(), batch['label']) + aux_loss

        if self.task in ['dti', 'moa']:
            probs = torch.softmax(out, dim=1)[:, 1]
            preds = torch.argmax(out, dim=1)
        loss = 0
        return loss, out, probs, preds
