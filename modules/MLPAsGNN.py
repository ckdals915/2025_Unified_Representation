'''
******************************************************************************
* @author   Handong Global University Industrial Intelligence Lab
* @Mod	    2025-07-15
* @brief    Unified Representation Linear Component as Message Passing
******************************************************************************
'''

import math
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class LinearAsGNN(MessagePassing):
    """
    A linear layer implemented via MessagePassing over a self-loop graph.
    Performs y = xW + b for each node.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bias: bool = True):
        super(LinearAsGNN, self).__init__(aggr='add')
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize weight and bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.size(1)  
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Support both 2D: (N, F) and 3D: (B, N, F)
        device = x.device
        if x.dim() == 3:
            B, N, F = x.shape
            
            # Flatten batch and nodes
            x_flat = x.view(B * N, F)
            
            # Build self-loop edges for B*N nodes
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_index, _ = add_self_loops(edge_index, num_nodes=B * N)
            
            # Message passing
            out_flat = self.propagate(edge_index.to(device), x=x_flat.to(device), size=(B * N, B * N))
            
            # Reshape back to (B, N, out_channels)
            out = out_flat.view(B, N, -1)
            return out
        
        elif x.dim() == 2:
            # Single-graph case: (N, F)
            N, F = x.shape
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)
            out = self.propagate(edge_index.to(device), x=x.to(device), size=(N, N))
            return out
        
        else:
            raise ValueError("Input tensor must be 2D or 3D")
    
    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        # x_j: [N, in_channels] for self-loop graph
        # W : [out_channels, in_channels] --> [in_channels, out_channels]
        out = x_j @ self.weight.t()
        if self.bias is not None:
            out += self.bias
        return out
    
    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        # aggr_out: [N, out_channels]
        return aggr_out


class MLPAsGNN(nn.Module):
    """
    A 2-layer MLP built from LinearAsGNN layers.
    Applies two GNN-linear transforms with activation in between.
    """
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 out_channels: int,
                 activation: str = 'relu', 
                 bias: bool = True,
                 dropout: float = 0.0,
                 device: torch.device = torch.device('cpu')
                 ):
        super(MLPAsGNN, self).__init__()
        self.linear1 = LinearAsGNN(in_channels, hidden_channels, bias=bias).to(device)
        self.linear2 = LinearAsGNN(hidden_channels, out_channels, bias=bias).to(device)
        
        # Activation function
        if isinstance(activation, str):
            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'gelu':
                self.activation = nn.GELU()
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
        else:
            self.activation = activation

        # Optional Dropout
        self.use_dropout = dropout > 0.0
        self.dropout = nn.Dropout(dropout) if self.use_dropout else None

        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.linear2(x)
        return x
    
        
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1) 배치 크기를 1로 설정
    B, D_in, D_hid, D_out = 1, 16, 32, 10
    num_iters = 10000

    # 2) 모델 정의 (한 번만)
    mlp = nn.Sequential(
        nn.Linear(D_in, D_hid),
        nn.ReLU(),
        nn.Linear(D_hid, D_out)
    ).to(device)

    gnn_mlp = MLPAsGNN(D_in, D_hid, D_out, activation='relu', device=device)

    # 3) 한 번만 weight 복사
    with torch.no_grad():
        gnn_mlp.linear1.weight.copy_(mlp[0].weight)
        gnn_mlp.linear1.bias.copy_(mlp[0].bias)
        gnn_mlp.linear2.weight.copy_(mlp[2].weight)
        gnn_mlp.linear2.bias.copy_(mlp[2].bias)

    total_diff = 0.0

    # 4) 10,000번 반복해서 MAE 누적
    for _ in range(num_iters):
        # 매번 새로운 난수 입력 생성
        x = torch.randn(B, D_in, device=device)

        # 두 모델에 넣어서 출력 계산
        y_ref = mlp(x)
        y_gnn = gnn_mlp(x)

        # 한 번 실행한 MAE (batch=1 이므로 사실상 |y_ref - y_gnn|의 평균)
        diff = (y_ref - y_gnn).abs().mean().item()
        total_diff += diff

    # 5) 최종 누적값과(원한다면) 평균값 계산
    avg_diff = total_diff / num_iters
    print(f"total_diff (10k iters): {total_diff:.6e}")
    print(f"avg_diff per iteration: {avg_diff:.6e}")
