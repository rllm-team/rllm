from __future__ import annotations
from typing import Optional, Union, Dict, List, Any

import torch
from torch import Tensor
from torch.nn import Parameter

from rllm.types import ColType
from rllm.nn.pre_encoder import FTTransformerPreEncoder


#class FTTransformerConv(torch.nn.Module):



#TODO 看一下transtab dataloader的逻辑
    

'''
# dataloader部分需要加上按顺序拼接「二元→数值→类别」的整体矩阵
def concat_all_features(td: TableData) -> Tensor:
    blocks = []
    for ctype in (ColType.BINARY, ColType.NUMERICAL, ColType.CATEGORICAL):
        if ctype in td.feat_dict:
            blocks.append(td.feat_dict[ctype])
    return torch.cat(blocks, dim=-1)
'''



#最初的版本
# class TransTabConv(nn.Module):
#     """
#     Convolutional wrapper for TransTab:
#     1) DataFrame -> tokens/values -> embeddings
#     2) prepend CLS token
#     3) stack of TransTabTransformerLayer
#     4) return CLS embedding 或 全序列 embeddings
    
#     Args:
#         conv_dim:           hidden dimension D
#         metadata:          metadata mapping (传给 PreEncoder)
#         vocab_size:        词表大小
#         padding_idx:       padding token id
#         feedforward_dim:   FFN 层维度 (默认 D)
#         num_heads:         多头数
#         num_layers:        Transformer 层数量
#         dropout:           dropout 概率
#         activation:        激活函数名 ("relu"/"gelu"/...)
#         use_cls:           是否只返回 CLS 向量
#         device:            设备
#         categorical_columns, numerical_columns, binary_columns:
#                            传给 DataExtractor 的列名单
#     """
#     def __init__(
#         self,
#         conv_dim: int,
#         metadata: Dict[ColType, List[Dict[str, Any]]],
#         vocab_size: int,
#         padding_idx: int = 0,
#         feedforward_dim: Optional[int] = None,
#         num_heads: int = 8,
#         num_layers: int = 2,
#         dropout: float = 0.1,
#         activation: str = "relu",
#         use_cls: bool = True,
#         device: Union[str, torch.device] = "cpu",
#         categorical_columns: Optional[List[str]] = None,
#         numerical_columns: Optional[List[str]] = None,
#         binary_columns: Optional[List[str]] = None,
#     ):
#         super().__init__()
#         self.device = torch.device(device)
#         self.use_cls = use_cls
#         self.conv_dim = conv_dim

#         # —— 1) PreEncoder + DataProcessor —— 
#         pre_encoder = TransTabPreEncoder(
#             out_dim=conv_dim,
#             metadata=metadata,
#             vocab_size=vocab_size,
#             padding_idx=padding_idx,
#             hidden_dropout_prob=dropout,
#             layer_norm_eps=1e-5,
#         ).to(self.device)

#         self.processor = TransTabDataProcessor(
#             pre_encoder=pre_encoder,
#             out_dim=conv_dim,
#             device=self.device,
#         )

#         # —— 2) CLS Token —— 
#         self.cls_token = TransTabCLSToken(hidden_dim=conv_dim).to(self.device)

#         # —— 3) Transformer Layers —— 
#         self.layers = nn.ModuleList()
#         for _ in range(num_layers):
#             layer = TransTabTransformerLayer(
#                 d_model=conv_dim,
#                 nhead=num_heads,
#                 dim_feedforward=feedforward_dim or conv_dim,
#                 dropout=dropout,
#                 activation=activation,
#                 layer_norm_eps=1e-5,
#                 batch_first=True,
#                 norm_first=False,
#                 use_layer_norm=True,
#             )
#             self.layers.append(layer)

#         self.reset_parameters()
#         self.to(self.device)

#     def reset_parameters(self):
#         # 初始化 CLS token
#         nn_init.uniform_(
#             self.cls_token.weight,
#             a=-1 / self.conv_dim**0.5,
#             b= 1 / self.conv_dim**0.5,
#         )
#         # 初始化 Transformer 参数
#         for layer in self.layers:
#             for p in layer.parameters():
#                 if p.dim() > 1:
#                     nn_init.xavier_uniform_(p)
#         # 初始化 PreEncoder 参数
#         self.processor.pre_encoder.reset_parameters()

#     def forward(
#         self,
#         df: Any,           # pandas.DataFrame
#         shuffle: bool = False,
#     ) -> Tensor:
#         """
#         Args:
#             df:      输入表格
#             shuffle: 是否在 extractor 阶段打乱列顺序
#         Returns:
#             如果 use_cls=True, 返回 [batch, conv_dim] 的 CLS 向量
#             否则返回 [batch, seq_len, conv_dim] 的完整序列
#         """
#         # ——— 提取 & 编码 ———
#         out = self.processor(df, shuffle=shuffle)
#         emb  = out["embedding"]       # [B, L, D]
#         mask = out["attention_mask"]   # [B, L]

#         # ——— 插入 CLS token ———
#         cls_out = self.cls_token(emb, mask)
#         x = cls_out["embedding"]       # [B, L+1, D]
#         attn = cls_out["attention_mask"]

#         # ——— Transformer 堆叠 ———
#         for layer in self.layers:
#             x = layer(x, src_key_padding_mask=attn)

#         # ——— 根据 use_cls 选输出 ———
#         if self.use_cls:
#             return x[:, 0, :]       # CLS
#         else:
#             return x[:, 1:, :]     # 去掉 CLS，返回主体
        






#加入考虑TableData的版本
# class TransTabDataProcessor(nn.Module):
#     """
#     Combine TransTabDataExtractor with TransTabPreEncoder,
#     now supports both pd.DataFrame and TableData as input.
#     """
#     def __init__(
#         self,
#         pre_encoder: TransTabPreEncoder,
#         out_dim: int,
#         device: Union[str, torch.device] = "cpu",
#     ) -> None:
#         super().__init__()
#         self.pre_encoder = pre_encoder.to(device)
#         # Initialize extractor without preset column lists; they will be set dynamically in forward
#         self.extractor = TransTabDataExtractor(
#             categorical_columns=None,
#             numerical_columns=None,
#             binary_columns=None,
#         )
#         # Alignment linear layer
#         self.align_layer = nn.Linear(out_dim, out_dim, bias=False).to(device)
#         self.device = device

#     def forward(
#         self,
#         data: Union[pd.DataFrame, TableData],
#         shuffle: bool = False,
#     ) -> Dict[str, torch.Tensor]:
#         # --- 1. Handle TableData input
#         if isinstance(data, TableData):
#             table = data
#             # Uncomment below to explicitly materialize features if needed
#             # table.lazy_materialize()
#             df = table.df
#             cat_cols = [c for c, t in table.col_types.items() if t == ColType.CATEGORICAL]
#             num_cols = [c for c, t in table.col_types.items() if t == ColType.NUMERICAL]
#             bin_cols = [c for c, t in table.col_types.items() if t == ColType.BINARY]
#             # Assign columns dynamically
#             self.extractor.categorical_columns = cat_cols or None
#             self.extractor.numerical_columns   = num_cols or None
#             self.extractor.binary_columns      = bin_cols or None
#         else:
#             df = data  # Already a DataFrame

#         # --- 2. Extract raw tensors
#         raw = self.extractor(df, shuffle=shuffle)

#         # --- 3. Build feature dict for PreEncoder
#         feat_dict: Dict[ColType, torch.Tensor] = {}
#         if raw["x_cat_input_ids"] is not None:
#             feat_dict[ColType.CATEGORICAL] = raw["x_cat_input_ids"].to(self.device)
#         if raw["x_bin_input_ids"] is not None:
#             feat_dict[ColType.BINARY] = raw["x_bin_input_ids"].to(self.device)
#         if raw["x_num"] is not None:
#             feat_dict[ColType.NUMERICAL] = (
#                 raw["num_col_input_ids"].to(self.device),
#                 raw["num_att_mask"].to(self.device),
#                 raw["x_num"].to(self.device),
#             )

#         # --- 4. Pass through PreEncoder to get embeddings
#         emb_dict = self.pre_encoder(feat_dict, return_dict=True)

#         # --- 5. Re-align embeddings
#         num_emb = emb_dict.get(ColType.NUMERICAL)
#         if num_emb is not None:
#             num_emb = self.align_layer(num_emb)
#         cat_emb = emb_dict.get(ColType.CATEGORICAL)
#         if cat_emb is not None:
#             cat_emb = self.align_layer(cat_emb)
#         bin_emb = emb_dict.get(ColType.BINARY)
#         if bin_emb is not None:
#             bin_emb = self.align_layer(bin_emb)

#         # --- 6. Concatenate embeddings and attention masks
#         emb_list, mask_list = [], []
#         if num_emb is not None:
#             emb_list.append(num_emb)
#             mask_list.append(torch.ones(
#                 num_emb.shape[0], num_emb.shape[1], device=self.device
#             ))
#         if cat_emb is not None:
#             emb_list.append(cat_emb)
#             mask_list.append(raw["cat_att_mask"].to(self.device).float())
#         if bin_emb is not None:
#             emb_list.append(bin_emb)
#             mask_list.append(raw["bin_att_mask"].to(self.device).float())

#         all_emb  = torch.cat(emb_list, dim=1)
#         all_mask = torch.cat(mask_list, dim=1)
#         return {"embedding": all_emb, "attention_mask": all_mask}