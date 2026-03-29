import torch
import torch.nn as nn
from torch.nn import functional as F
import functools


def get_norm_layer(norm_type='none'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm1d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm1d, affine=False, track_running_stats=False)
    elif norm_type == 'layer':
        norm_layer = functools.partial(nn.LayerNorm)
    elif norm_type == 'none':
        norm_layer = functools.partial(nn.Identity)
    else:
        raise NotImplementedError(f'normalization layer [{norm_type}] is not found')
    return norm_layer


class TreeGateBranchingNet(nn.Module):
    def __init__(self, branch_size, tree_state_size, dim_reduce_factor=2, infimum=8, norm='layer', depth=2, hidden_size=128):
        super().__init__()
        # 修复版：精简网络深度，避免小维度 LayerNorm 和指数级梯度消失
        
        # 1. 变量特征提取层 (保持在安全的大维度 hidden_size)
        self.feature_net = nn.Sequential(
            nn.Linear(branch_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 2. Tree Gate 生成器 (生成一层的全局注意力权重)
        self.gate_net = nn.Sequential(
            nn.Linear(tree_state_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid() # 限制在 0-1 之间作为门控开关
        )
        
        # 3. 最终打分层 (平滑降维输出标量分数，不加 LayerNorm)
        self.scoring_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, cands_state_mat, tree_state):
        # cands_state_mat: [B, L, H]
        # 1. 提炼变量特征
        cands_feat = self.feature_net(cands_state_mat)
        
        # 2. 生成 Tree Gate: [B, H] -> [B, 1, H]
        gate = self.gate_net(tree_state).unsqueeze(1)
        
        # 3. Tree Gating 核心机制：用当前的树状态去“过滤/激活”候选变量特征
        gated_cands = (cands_feat * gate) +  cands_feat
        
        # 4. 独立打分: [B, L, 1] -> [B, L]
        scores = self.scoring_layer(gated_cands).squeeze(-1)
        
        return scores


class BiMatchingNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1_1 = nn.Linear(hidden_size, hidden_size)
        self.linear1_2 = nn.Linear(hidden_size, hidden_size)
        self.linear2_1 = nn.Linear(hidden_size, hidden_size)
        self.linear2_2 = nn.Linear(hidden_size, hidden_size)
        self.linear3_1 = nn.Linear(hidden_size, hidden_size)
        self.linear3_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, tree_feat, var_feat, padding_mask):
        """
        tree_feat: [B, E]
        var_feat:  [B, L, E]
        padding_mask: [B, L] True for pads; may be None
        """
        B, L, E = var_feat.shape
        if padding_mask is None:
            padding_mask = torch.zeros((B, L), dtype=torch.bool, device=var_feat.device)

        tree_feat = tree_feat.unsqueeze(1)  # [B,1,E]
        G_tc = torch.bmm(self.linear1_1(tree_feat), var_feat.transpose(1, 2)).squeeze(1)  # [B, L]
        G_tc = G_tc.masked_fill(padding_mask, float('-inf'))
        G_tc = F.softmax(G_tc, dim=1).unsqueeze(1)  # [B,1,L]

        G_ct = torch.bmm(self.linear1_2(var_feat), tree_feat.transpose(1, 2)).squeeze(2)  # [B,L]
        G_ct = G_ct.masked_fill(padding_mask, float('-inf'))
        G_ct = F.softmax(G_ct, dim=1).unsqueeze(2)  # [B,L,1]

        E_t = torch.bmm(G_tc, var_feat)              # [B,1,E]
        E_c = torch.bmm(G_ct, tree_feat)             # [B,L,1] x [B,1,E] -> [B,L,E]

        S_tc = F.relu(self.linear2_1(E_t))           # [B,1,E]
        S_ct = F.relu(self.linear2_2(E_c))           # [B,L,E]

        attn_weight = torch.sigmoid(self.linear3_1(S_tc) + self.linear3_2(S_ct))  # [B,L,E]
        M_tc = attn_weight * S_tc + (1 - attn_weight) * S_ct
        out_feat = var_feat + M_tc

        return out_feat