'''
******************************************************************************
* @author   Handong Global University Industrial Intelligence Lab
* @Mod	    2025-07-15
* @brief    Unified Representation Pooling Component as Message Passing
******************************************************************************
'''

import math
import torch
from torch import nn
from torch_geometric.nn import MessagePassing


class AvgPool1dAsGNN(MessagePassing):
    """
    Implements a 1D Average Pooling as a graph neural network layer using PyTorch Geometric's MessagePassing.
    Supports 'valid' (MUST no pad) modes.
    - aggr='mean' : calculate features(NOT mean aggregator)
    - update() : divide kernel_size
    """

    def __init__(self,
                 length: int,
                 kernel_size: int,
                 stride: int
                 ):
        # use add aggregation
        super().__init__(aggr='mean')

        self.length = length
        self.kernel_size = kernel_size
        self.stride = stride

        # compute output dims
        self.out_length = math.floor((length - kernel_size) / stride) + 1

        self.num_in_nodes = length
        self.num_out_nodes = self.out_length

        # build static adjacency
        src, dst = self._build_adjacency()
        self.register_buffer('edge_idx', torch.tensor([src, dst], dtype=torch.long))

    # Make Adjacency Matrix
    def _build_adjacency(self):
        src_list, dst_list = [], []
        centers = [i * self.stride for i in range(self.out_length)]

        for out_idx, center in enumerate(centers):
            for offset in range(self.kernel_size):
                inp = center + offset
                src_list.append(inp)
                dst_list.append(out_idx)

        return src_list, dst_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L] or [C, L]
        is_batch = (x.dim() == 3)
        if is_batch:
            B, C, L = x.shape
            # flatten all pixels across batch
            x_flat = x.permute(0, 2, 1).reshape(-1, C)

            # replicate edge_idx for each batch element
            num_in, num_out = self.num_in_nodes, self.num_out_nodes
            src, dst = self.edge_idx
            
            # make lists of offset indices
            src_batches = [src + b * num_in for b in range(B)]
            dst_batches = [dst + b * num_out for b in range(B)]
            batched_edge_idx = torch.stack([
                torch.cat(src_batches, dim=0),
                torch.cat(dst_batches, dim=0)
            ], dim=0)

            # Message Passing
            out = self.propagate(
                batched_edge_idx,
                x=x_flat,
                size=(B * num_in, B * num_out)
            )

            # reshape: [B * num_out, C] → [B, out_H, out_W, C]
            out = out.view(B, self.out_length, C)
            return out.permute(0, 2, 1)

        else:
            # Single Graph
            C, L = x.shape
            x_flat = x.permute(1, 0).reshape(-1, C)
            
            # Message Passing
            out = self.propagate(
                self.edge_idx,
                x=x_flat,
                size=(self.num_in_nodes, self.num_out_nodes)
            )
            
            # Reshape output
            out = out.view(self.out_length, C)
            return out.permute(1, 0)

    # Message Function(Identity)
    def message(self, x_j):
        return x_j

    # Update Function(Identity)
    def update(self, aggr_out):
        return aggr_out


class AvgPool2dAsGNN(MessagePassing):
    """
    Implements a 2D Average Pooling as a graph neural network layer using PyTorch Geometric's MessagePassing.
    Supports 'valid' (MUST no pad) modes.
    - aggr='mean' : calculate features(NOT mean aggregator)
    - update() : divide kernel_size**2
    """

    def __init__(self,
                 height: int,
                 width: int,
                 kernel_size: int,
                 stride: int
                 ):
        # use mean aggregation
        super().__init__(aggr='mean')

        self.height = height
        self.width = width
        self.kernel_size = kernel_size
        self.stride = stride

        # compute output dims
        self.out_height = math.floor((height - kernel_size) / stride) + 1
        self.out_width = math.floor((width - kernel_size) / stride) + 1

        # Number of Input and Output Nodes
        self.num_in_nodes = height * width
        self.num_out_nodes = self.out_height * self.out_width

        # Make Adjacency Matrix
        src, dst = self._build_adjacency()
        self.register_buffer('edge_idx', torch.tensor([src, dst], dtype=torch.long))

    # Make Adjacency Matrix
    def _build_adjacency(self):
        src_list, dst_list = [], []

        rows = [i * self.stride for i in range(self.out_height)]
        cols = [i * self.stride for i in range(self.out_width)]

        for out_r, r in enumerate(rows):
            for out_c, c in enumerate(cols):
                dst = out_r * self.out_width + out_c
                for dr in range(self.kernel_size):
                    for dc in range(self.kernel_size):
                        src = (r + dr) * self.width + (c + dc)
                        src_list.append(src)
                        dst_list.append(dst)
        return src_list, dst_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Batch Graph
        is_batch = (x.dim() == 4)
        if is_batch:
            B, C, H, W = x.shape
            # Flatten all pixels across batch
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)

            # replicate edge_idx for each batch element
            num_in, num_out = self.num_in_nodes, self.num_out_nodes
            src, dst = self.edge_idx 
            
            # make lists of offset indices
            src_batches = [src + b * num_in for b in range(B)]
            dst_batches = [dst + b * num_out for b in range(B)]
            batched_edge_idx = torch.stack([
                torch.cat(src_batches, dim=0),
                torch.cat(dst_batches, dim=0)
            ], dim=0)  

            # Message Passing
            out = self.propagate(
                batched_edge_idx,
                x=x_flat,
                size=(B * num_in, B * num_out)
            )

            # reshape: [B * num_out, C] → [B, out_H, out_W, C]
            out = out.view(B, self.out_height, self.out_width, C)
            return out.permute(0, 3, 1, 2)

        else:
            # Single Graph
            C, H, W = x.shape
            x_flat = x.permute(1, 2, 0).reshape(-1, C)
            
            # Message Passing
            out = self.propagate(
                self.edge_idx, x=x_flat,
                size=(self.num_in_nodes, self.num_out_nodes)
            )
            
            # Reshape output
            out = out.view(self.out_height, self.out_width, C)
            return out.permute(2, 0, 1)

    # Message Function(Identity)
    def message(self, x_j):
        return x_j

    # Update Function(Identity)
    def update(self, aggr_out):
        return aggr_out

class MaxPool1dAsGNN(MessagePassing):
    """
    Implements a 1D MaxPool as a graph neural network layer using PyTorch Geometric's MessagePassing.
    Supports 'same' (zero‐padded) and 'valid' (no pad) modes.
    """

    def __init__(self,
                 length: int,
                 kernel_size: int,
                 stride: int,
                 padding: str = 'valid'):
        # use max aggregation
        super().__init__(aggr='max')

        assert padding in ('same', 'valid')
        self.length = length
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # compute output length
        if padding == 'same':
            self.out_length = math.ceil(length / stride)
        else:
            self.out_length = math.floor((length - kernel_size) / stride) + 1

        # Number of Input and Output Nodes
        self.num_in_nodes = length
        self.num_out_nodes = self.out_length

        # Make Adjacency Matrix
        src, dst = self._build_adjacency()
        self.register_buffer('edge_idx', torch.tensor([src, dst], dtype=torch.long))

    # Make Adjacency Matrix
    def _build_adjacency(self):
        src_list, dst_list = [], []

        pad = self.kernel_size // 2
        if self.padding == 'same':
            centers = [i * self.stride - pad for i in range(self.out_length)]
        else:
            centers = [i * self.stride for i in range(self.out_length)]

        for out_idx, center in enumerate(centers):
            for offset in range(self.kernel_size):
                inp = center + offset
                if self.padding == 'same' and not (0 <= inp < self.length):
                    continue
                src_list.append(inp)
                dst_list.append(out_idx)

        return src_list, dst_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        is_batch = (x.dim() == 3)
        if is_batch:
            # Batch Graph
            B, C, L = x.shape
            
            # flatten all pixels across batch
            x_flat = x.permute(0, 2, 1).reshape(-1, C)

            # replicate edge_idx for each batch element
            num_in, num_out = self.num_in_nodes, self.num_out_nodes
            src, dst = self.edge_idx
            
            # Make Node Index using offset
            src_batches = [src + b * num_in for b in range(B)]
            dst_batches = [dst + b * num_out for b in range(B)]
            
            # Make Batched Edge Index
            batched_edge_idx = torch.stack([
                torch.cat(src_batches, dim=0),
                torch.cat(dst_batches, dim=0)
            ], dim=0)

            # Message Passing
            out = self.propagate(
                batched_edge_idx,
                x=x_flat,
                size=(B * num_in, B * num_out)
            )
            
            # reshape: [B * num_out, C] → [B, out_H, out_W, C]
            out = out.view(B, self.out_length, C)
            return out.permute(0, 2, 1)

        else:
            # Single Graph
            C, L = x.shape
            x_flat = x.permute(1, 0).reshape(-1, C)
            
            # Message Passing
            out = self.propagate(
                self.edge_idx,
                x=x_flat,
                size=(self.num_in_nodes, self.num_out_nodes)
            )
            
            # Reshape output
            out = out.view(self.out_length, C)
            return out.permute(1, 0)


class MaxPool2dAsGNN(MessagePassing):
    """
    Implements a 2D MaxPool as a graph neural network layer using PyTorch Geometric's MessagePassing.
    Supports 'same' (zero‐padded) and 'valid' (no pad) modes.
    """
    def __init__(self,
                 height: int,
                 width: int,
                 kernel_size: int,
                 stride: int,
                 padding: str = 'valid'):
        # use max aggregation
        super().__init__(aggr='max')

        assert padding in ('same', 'valid')
        self.height = height
        self.width = width
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # compute output dims
        if padding == 'same':
            self.out_height = math.ceil(height / stride)
            self.out_width  = math.ceil(width  / stride)
        else: 
            self.out_height = math.floor((height - kernel_size) / stride) + 1
            self.out_width  = math.floor((width  - kernel_size) / stride) + 1

        self.num_in_nodes  = height * width
        self.num_out_nodes = self.out_height * self.out_width

        # Make Adjacency Matrix
        src, dst = self._build_adjacency()
        self.register_buffer('edge_idx', torch.tensor([src, dst], dtype=torch.long))

    # Make Adjacency Matrix
    def _build_adjacency(self):
        src_list, dst_list = [], []

        pad = self.kernel_size // 2
        if self.padding == 'same':
            rows = [i * self.stride - pad for i in range(self.out_height)]
            cols = [j * self.stride - pad for j in range(self.out_width)]
        else:
            rows = [i * self.stride for i in range(self.out_height)]
            cols = [j * self.stride for j in range(self.out_width)]

        for out_r, r in enumerate(rows):
            for out_c, c in enumerate(cols):
                dst = out_r * self.out_width + out_c
                for dr in range(self.kernel_size):
                    for dc in range(self.kernel_size):
                        in_r = r + dr
                        in_c = c + dc
                        if self.padding == 'same' and not (0 <= in_r < self.height and 0 <= in_c < self.width):
                            continue
                        src = in_r * self.width + in_c
                        src_list.append(src)
                        dst_list.append(dst)

        return src_list, dst_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_batch = (x.dim() == 4)
        if is_batch:
            B, C, H, W = x.shape
            # Flatten all pixels across batch
            x_flat = x.permute(0,2,3,1).reshape(-1, C)  # [B * num_in, C]

            # replicate edge_idx for each batch element
            num_in, num_out = self.num_in_nodes, self.num_out_nodes
            src, dst = self.edge_idx
            
            # make lists of offset index
            src_batches = [src + b * num_in for b in range(B)]
            dst_batches = [dst + b * num_out for b in range(B)]
            
            # Make Batched Edge Index
            batched_edge_idx = torch.stack([
                torch.cat(src_batches, dim=0),
                torch.cat(dst_batches, dim=0)
            ], dim=0)

            # Message Passing
            out = self.propagate(
                batched_edge_idx, 
                x=x_flat,
                size=(B * num_in, B * num_out)
            )
            
            # reshape: [B * num_out, C] → [B, out_H, out_W, C]
            out = out.view(B, self.out_height, self.out_width, C)
            return out.permute(0,3,1,2)

        else:
            # Single Graph
            C, H, W = x.shape
            
            # Flatten all pixels across batch
            x_flat = x.permute(1,2,0).reshape(-1, C)
            
            # Message Passing
            out = self.propagate(
                self.edge_idx, x=x_flat,
                size=(self.num_in_nodes, self.num_out_nodes)
            )
            
            # Reshape output
            out = out.view(self.out_height, self.out_width, C)
            return out.permute(2,0,1)
        
        
if __name__ == "__main__":
    # seed
    torch.manual_seed(10)

    # '''
    # MaxPool1d
    # '''
    # # Configuration
    # L, Cin, K, S = 64, 3, 5, 2

    # # Random input (batch size=10, positive values only)
    # x_seq = torch.rand(10, Cin, L)

    # # Test both padding modes
    # for padding in ['same', 'valid']:
    #     # GNN-based MaxPool1d
    #     gnn_pool = MaxPool1dAsGNN(
    #         length=L,
    #         kernel_size=K,
    #         stride=S,
    #         padding=padding
    #     )
        
    #     # Standard nn.MaxPool1d
    #     if padding == 'same':
    #         pool = nn.MaxPool1d(K, stride=S, padding=K // 2)
    #     else:
    #         pool = nn.MaxPool1d(K, stride=S, padding=0)

    #     # Forward pass
    #     y_gnn = gnn_pool(x_seq) 
    #     y_pool = pool(x_seq) 

    #     # Difference calculation
    #     diff = torch.abs(y_gnn - y_pool)
    #     max_diff = diff.mean().item()

    #     print(f"=== padding={padding!r} ===")
    #     print(f" GNN output shape : {tuple(y_gnn.shape)}")
    #     print(f" Pool1d output shape: {tuple(y_pool.shape)}")
    #     print(f" mean abs difference : {max_diff:.3e}\n")

    # '''
    # MaxPool2d
    # '''
    # # Configuration 
    # W, H, Cin, K, S = 32, 32, 3, 5, 2

    # # Random input (batch size=10, positive values only)
    # x_img = torch.rand(10, Cin, H, W)

    # for padding in ['same', 'valid']:
    #     # GNN-based MaxPool2d
    #     gnn_pool = MaxPool2dAsGNN(
    #         width=W,
    #         height=H,
    #         kernel_size=K,
    #         stride=S,
    #         padding=padding
    #     )
        
    #     # Standard nn.MaxPool2d
    #     if padding == 'same':
    #         pool = nn.MaxPool2d(K, stride=S, padding=K // 2)
    #     else:
    #         pool = nn.MaxPool2d(K, stride=S, padding=0)

    #     # Forward pass
    #     y_gnn = gnn_pool(x_img)
    #     y_pool = pool(x_img)

    #     # Difference calculation
    #     diff = torch.abs(y_gnn - y_pool)
    #     max_diff = diff.mean().item()

    #     print(f"=== padding={padding!r} ===")
    #     print(f" GNN output shape : {tuple(y_gnn.shape)}")
    #     print(f" Pool2d output shape: {tuple(y_pool.shape)}")
    #     print(f" mean abs difference : {max_diff:.3e}\n")

    # '''
    # AvgPool1d
    # '''
    # # Configuration
    # L, Cin, K, S = 64, 3, 5, 2

    # # Random input (batch size=10, positive values only)
    # x_seq = torch.rand(10, Cin, L)

    # # GNN-based AvgPool1d
    # gnn_pool = AvgPool1dAsGNN(
    #     length=L,
    #     kernel_size=K,
    #     stride=S,
    # )
    # # Standard nn.AvgPool1d
    # pool = nn.AvgPool1d(K, stride=S, padding=0)

    # # Forward pass
    # y_gnn = gnn_pool(x_seq) 
    # y_pool = pool(x_seq)

    # # Difference calculation
    # diff = torch.abs(y_gnn - y_pool)
    # max_diff = diff.mean().item()

    # print(f" GNN output shape : {tuple(y_gnn.shape)}")
    # print(f" AvgPool1d output shape: {tuple(y_pool.shape)}")
    # print(f" Mean abs difference : {max_diff:.3e}\n")

    # '''
    # AvgPool2d
    # '''
    # # Configuration
    # W, H, Cin, K, S = 32, 32, 3, 5, 2

    # # Random input (batch size=10, positive values only)
    # x_img = torch.rand(10, Cin, H, W)

    # # GNN-based AvgPool2d
    # gnn_pool = AvgPool2dAsGNN(
    #     width=W,
    #     height=H,
    #     kernel_size=K,
    #     stride=S
    # )
    
    # # Standard nn.AvgPool2d
    # pool = nn.AvgPool2d(K, stride=S, padding=0)

    # # Forward pass
    # y_gnn = gnn_pool(x_img)  
    # y_pool = pool(x_img) 

    # # Difference calculation
    # diff = torch.abs(y_gnn - y_pool)
    # max_diff = diff.mean().item()

    # print(f" GNN output shape : {tuple(y_gnn.shape)}")
    # print(f" Pool2d output shape: {tuple(y_pool.shape)}")
    # print(f" mean abs difference : {max_diff:.3e}\n")
    
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ######################## MaxPool2D ########################
    num_iters = 10000  # 반복 횟수
    B = 1  # 배치 크기
    Cin = 3  # 입력 채널 수
    H, W = 32, 32  # 입력 이미지 크기
    ks_pairs = [(2, 2), (3, 2)]

    for (K, S) in ks_pairs:
        print(f"\n=== MaxPool2D 비교: K={K}, S={S} ===")

        for padding in [ 'valid']:
            # 1) GNN 기반 MaxPool2D 생성
            gnn_pool2d = MaxPool2dAsGNN(
                height=H,
                width=W,
                kernel_size=K,
                stride=S,
                padding=padding
            ).to(device)

            # 2) 표준 nn.MaxPool2d 생성
            if padding == 'same':
                pool2d = nn.MaxPool2d(K, stride=S, padding=K // 2).to(device)
            else:
                pool2d = nn.MaxPool2d(K, stride=S, padding=0).to(device)

            # 3) MAE 누적 계산
            total_diff = 0.0
            for _ in range(num_iters):
                x_img = torch.randn(B, Cin, H, W, device=device)
                y_gnn = gnn_pool2d(x_img)
                y_std = pool2d(x_img)
                diff = (y_gnn - y_std).abs().mean().item()
                total_diff += diff

            # 4) 출력 형태 한 번 확인
            x_sample = torch.randn(B, Cin, H, W, device=device)
            out_shape_gnn = tuple(gnn_pool2d(x_sample).shape)
            out_shape_std = tuple(pool2d(x_sample).shape)
            avg_mae = total_diff / num_iters

            print(f"padding={padding!r} → Output shapes: GNN{out_shape_gnn}, Std{out_shape_std}")
            print(f"MAE 누적 합계: {total_diff:.6e}, 회당 평균 MAE: {avg_mae:.6e}")

    ######################## MaxPool1D ########################
    num_iters = 10000
    B = 1
    Cin = 3
    L = 64  # 입력 시퀀스 길이
    ks_pairs_1d = [(2, 2), (3, 2)]

    for (K, S) in ks_pairs_1d:
        print(f"\n=== MaxPool1D 비교: K={K}, S={S} ===")

        for padding in [ 'valid']:
            # 1) GNN 기반 MaxPool1D 생성
            gnn_pool1d = MaxPool1dAsGNN(
                length=L,
                kernel_size=K,
                stride=S,
                padding=padding
            ).to(device)

            # 2) 표준 nn.MaxPool1d 생성
            if padding == 'same':
                pool1d = nn.MaxPool1d(K, stride=S, padding=K // 2).to(device)
            else:
                pool1d = nn.MaxPool1d(K, stride=S, padding=0).to(device)

            # 3) MAE 누적 계산
            total_diff = 0.0
            for _ in range(num_iters):
                x_seq = torch.randn(B, Cin, L, device=device)
                y_gnn = gnn_pool1d(x_seq)
                y_std = pool1d(x_seq)
                diff = (y_gnn - y_std).abs().mean().item()
                total_diff += diff

            # 4) 출력 형태 한 번 확인
            x_sample = torch.randn(B, Cin, L, device=device)
            out_shape_gnn = tuple(gnn_pool1d(x_sample).shape)
            out_shape_std = tuple(pool1d(x_sample).shape)
            avg_mae = total_diff / num_iters

            print(f"padding={padding!r} → Output shapes: GNN{out_shape_gnn}, Std{out_shape_std}")
            print(f"MAE 누적 합계: {total_diff:.6e}, 회당 평균 MAE: {avg_mae:.6e}")

    ######################## AvgPool2D ########################
    num_iters = 10000
    B = 1
    Cin = 3
    H, W = 32, 32
    ks_pairs = [(2, 2), (3, 2)]

    for (K, S) in ks_pairs:
        print(f"\n=== AvgPool2D 비교: K={K}, S={S} ===")

        for padding in ['valid']:  # AvgPool 일반적으로 'same' padding을 지원하지 않으므로 'valid'만 테스트
            # 1) GNN 기반 AvgPool2D 생성
            gnn_pool2d = AvgPool2dAsGNN(
                height=H,
                width=W,
                kernel_size=K,
                stride=S
            ).to(device)

            # 2) 표준 nn.AvgPool2d 생성
            pool2d = nn.AvgPool2d(K, stride=S, padding=0).to(device)

            # 3) MAE 누적 계산
            total_diff = 0.0
            for _ in range(num_iters):
                x_img = torch.randn(B, Cin, H, W, device=device)
                y_gnn = gnn_pool2d(x_img)
                y_std = pool2d(x_img)
                diff = (y_gnn - y_std).abs().mean().item()
                total_diff += diff

            # 4) 출력 형태 한 번 확인
            x_sample = torch.randn(B, Cin, H, W, device=device)
            out_shape_gnn = tuple(gnn_pool2d(x_sample).shape)
            out_shape_std = tuple(pool2d(x_sample).shape)
            avg_mae = total_diff / num_iters

            print(f"padding={padding!r} → Output shapes: GNN{out_shape_gnn}, Std{out_shape_std}")
            print(f"MAE 누적 합계: {total_diff:.6e}, 회당 평균 MAE: {avg_mae:.6e}")

    ######################## AvgPool1D ########################
    num_iters = 10000
    B = 1
    Cin = 3
    L = 64
    ks_pairs_1d = [(2, 2), (3, 2)]

    for (K, S) in ks_pairs_1d:
        print(f"\n=== AvgPool1D 비교: K={K}, S={S} ===")

        # AvgPool1D는 padding='valid'만 지원한다고 가정
        gnn_pool1d = AvgPool1dAsGNN(
            length=L,
            kernel_size=K,
            stride=S
        ).to(device)

        pool1d = nn.AvgPool1d(K, stride=S, padding=0).to(device)

        total_diff = 0.0
        for _ in range(num_iters):
            x_seq = torch.randn(B, Cin, L, device=device)
            y_gnn = gnn_pool1d(x_seq)
            y_std = pool1d(x_seq)
            diff = (y_gnn - y_std).abs().mean().item()
            total_diff += diff

        x_sample = torch.randn(B, Cin, L, device=device)
        out_shape_gnn = tuple(gnn_pool1d(x_sample).shape)
        out_shape_std = tuple(pool1d(x_sample).shape)
        avg_mae = total_diff / num_iters

        print(f"Output shapes: GNN{out_shape_gnn}, Std{out_shape_std}")
        print(f"MAE 누적 합계: {total_diff:.6e}, 회당 평균 MAE: {avg_mae:.6e}")