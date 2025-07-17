'''
******************************************************************************
* @author   Handong Global University Industrial Intelligence Lab
* @Mod	    2025-07-15
* @brief    Unified Representation Convolution Component as Message Passing
******************************************************************************
'''

import math
import torch
from torch import nn
from torch_geometric.nn import MessagePassing

class Conv1dAsGNN(MessagePassing):
    """
    Implements a 1D Convolutional GNN message passing layer
    """

    def __init__(self,
                 length: int,
                 kernel_size: int,
                 stride: int,
                 in_channels: int,
                 out_channels: int,
                 padding: str = 'same',
                 bias: bool = True
                 ):
        # Sum Aggregation Function
        super(Conv1dAsGNN, self).__init__(aggr='add')
        assert padding in ('same', 'valid')
        self.length = length
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        # Padding Calculation : 짝수 커널일 때 추가 처리 필요
        if padding == 'same':
            self.out_length = math.ceil(length / stride)
            total_pad = kernel_size - 1
            self.pad_left = total_pad // 2
            self.pad_right = total_pad - self.pad_left
        
        # 'valid' padding does not require additional padding
        elif padding == 'valid':
            self.out_length = math.floor((length - kernel_size) / stride) + 1
            self.pad_left = self.pad_right = 0

        # Number of input and output nodes
        self.num_in_nodes = length
        self.num_out_nodes = self.out_length

        # Learnable weights: (in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(
            torch.Tensor(in_channels, out_channels, kernel_size)
        )

        # Bias term
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

        # Build static adjacency
        src, dst, ic_idx, p_idx = self._build_adjacency()
        
        # Edge index, input-channel indices, and position indices
        self.register_buffer('edge_idx', torch.tensor([src, dst], dtype=torch.long))
        self.register_buffer('ic_idx', torch.tensor(ic_idx, dtype=torch.long))
        self.register_buffer('p_idx', torch.tensor(p_idx, dtype=torch.long))

    # Function to reset parameters
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    # Function to build adjacency for the convolution operation
    def _build_adjacency(self):
        """
        Constructs lists of source indices, target indices, input-channel indices, and kernel-position indices
        for each valid kernel connection between input and output pixels.
        src_list: Neighboring input nodes
        dst_list: Output nodes
        ic_list: Input channel indices
        p_list: Kernel position indices
        """
        K = self.kernel_size
        S = self.stride
        L = self.length
        out_L = self.out_length
        src_list, dst_list, ic_list, p_list = [], [], [], []

        for ic in range(self.in_channels):
            for out_i in range(out_L):
                # Calculate the starting index for the kernel in the input sequence
                # e.g. Conv1d padding='same' → start = out_i*stride - pad_left
                start = out_i * S - self.pad_left
                dst = out_i
                p = 0
                
                # d = Calculate actual in_idx as it moves by 0..K-1
                for d in range(K):
                    in_idx = start + d
                    
                    # Padding Mode
                    if self.padding == 'same':
                        if not (0 <= in_idx < L):
                            p += 1
                            continue
                    
                    # Valid Mode
                    else:
                        if not (0 <= in_idx < L):
                            raise RuntimeError(
                                f"Invalid kernel region in 'valid' mode: "
                                f"out_i={out_i}, start={start}, d={d}, in_idx={in_idx}"
                            )

                    # Connect Edge
                    src_list.append(in_idx)
                    dst_list.append(dst)
                    ic_list.append(ic)
                    p_list.append(p)
                    p += 1

        return src_list, dst_list, ic_list, p_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
       
        # Convert edge_idx, ic_idx, p_idx to device
        edge_idx = self.edge_idx.to(device)  # (2, num_edges)
        ic_idx = self.ic_idx.to(device)  # (num_edges)
        p_idx = self.p_idx.to(device)  # (num_edges)

        # single sequence: (C, L)
        if x.dim() == 2:
            C, L = x.shape
            x_flat = x.permute(1, 0).reshape(-1, C).to(device)  # (L, C)
            out = self.propagate(
                edge_idx, x=x_flat,
                ic_idx=ic_idx, p_idx=p_idx,
                size=(self.num_in_nodes, self.num_out_nodes)
            )
            if self.bias is not None:
                out = out + self.bias
            out = out.view(self.out_length, self.out_channels)
            return out.permute(1, 0)  # (out_ch, out_len)

        # batched: (B, C, L)
        elif x.dim() == 3:
            B, C, L = x.shape
            x_flat = x.permute(0, 2, 1).reshape(-1, C).to(device)  # (B*L, C)

            # batch-aware edge_idx
            num_in, num_out = self.num_in_nodes, self.num_out_nodes
            src, dst = self.edge_idx
            src_batches = [src + b * num_in for b in range(B)]
            dst_batches = [dst + b * num_out for b in range(B)]
            batched_edge_idx = torch.stack([
                torch.cat(src_batches, dim=0),
                torch.cat(dst_batches, dim=0)
            ], dim=0).to(device)

            # replicate indices
            ic = self.ic_idx.unsqueeze(0).repeat(B, 1).view(-1).to(device)
            p = self.p_idx.unsqueeze(0).repeat(B, 1).view(-1).to(device)

            out_flat = self.propagate(
                batched_edge_idx,
                x=x_flat,
                ic_idx=ic,
                p_idx=p,
                size=(B * num_in, B * num_out)
            )
            if self.bias is not None:
                out_flat = out_flat + self.bias.to(device)

            out = out_flat.view(B, self.out_length, self.out_channels)
            return out.permute(0, 2, 1)  # (B, out_ch, out_len)

        else:
            raise ValueError("Input must be 2D or 3D tensor")

    # Message Function(dot product)
    def message(self, x_j: torch.Tensor, ic_idx: torch.Tensor, p_idx: torch.Tensor) -> torch.Tensor:
        x_scalar = x_j[torch.arange(x_j.size(0), device=x_j.device), ic_idx]
        edge_weight = self.weight[ic_idx, :, p_idx]
        return edge_weight * x_scalar.unsqueeze(-1)

    # Update Function(Identity)
    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return aggr_out


class Conv2dAsGNN(MessagePassing):
    """
    Implements a 2D convolution as a graph neural network layer using PyTorch Geometric's MessagePassing.
    Supports 'same' (zero-padded) and 'valid' (no pad) modes.
    src_list: Neighboring input nodes
    dst_list: Output nodes
    ic_list: Input channel indices
    p_list: Kernel position indices
    """

    def __init__(self,
                 height: int,
                 width: int,
                 kernel_size: int,
                 stride: int,
                 in_channels: int,
                 out_channels: int,
                 padding: str = 'same',
                 bias: bool = True
                 ):
        
        # Sum Aggregation Function
        super().__init__(aggr='add')
        assert padding in ('same', 'valid'), "Padding must be 'same' or 'valid'."
        
        self.height = height
        self.width = width
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

       
        
        # Padding 이전버전
        # if padding == 'same':
        #     self.out_height = math.ceil(height / stride)
        #     self.out_width = math.ceil(width / stride)
        #
        # elif padding == 'valid':
        #     self.out_height = math.floor((height - kernel_size) / stride) + 1
        #     self.out_width = math.floor((width - kernel_size) / stride) + 1

        # Padding Calculation : 짝수 커널일 때 추가 처리 필요
        if padding == 'same':
            p = kernel_size // 2
            self.pad_top = p
            self.pad_bottom = p
            self.pad_left = p
            self.pad_right = p
            # total_pad_h = kernel_size - 1
            # total_pad_w = kernel_size - 1
            # # e.g. kernel_size=8 → total_pad_h=7 → pad_top=3, pad_bottom=4
            # self.pad_top = total_pad_h // 2
            # self.pad_bottom = total_pad_h - self.pad_top
            # self.pad_left = total_pad_w // 2
            # self.pad_right = total_pad_w - self.pad_left
        else: 
            self.pad_top = self.pad_bottom = 0
            self.pad_left = self.pad_right = 0

        # Padding Output Feature Map Size
        if padding == 'same':
            self.out_height = math.floor((height + 2 * p - kernel_size) / stride) + 1
            self.out_width = math.floor((width + 2 * p - kernel_size) / stride) + 1

        else:
            self.out_height = math.floor((height - kernel_size) / stride) + 1
            self.out_width = math.floor((width - kernel_size) / stride) + 1

        # Number of input and output nodes
        self.num_in_nodes = height * width
        self.num_out_nodes = self.out_height * self.out_width

        # Learnable weights: (in_channels, out_channels, kernel_size**2)
        self.weight = nn.Parameter(
            torch.Tensor(in_channels, out_channels, kernel_size * kernel_size)
        )

        # Bias term
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

        # Build static adjacency once and register as buffers
        src, dst, ic_idx, p_idx = self._build_adjacency()
        self.register_buffer('edge_idx', torch.tensor([src, dst], dtype=torch.long))
        self.register_buffer('ic_idx', torch.tensor(ic_idx, dtype=torch.long))
        self.register_buffer('p_idx', torch.tensor(p_idx, dtype=torch.long))

    # Function to reset parameters
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * (self.kernel_size ** 2)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def _build_adjacency(self):
        """
        Constructs (src_list, dst_list, ic_list, p_list) for ANY kernel_size both even/odd kernels
        - padding='same' : it skips when the window goes out of the image, but p keeps increasing to ensure the total number of p_idxes: kernel_size* kernel_size.
        - padding='valid': only the window is allowed to be completely heard.
        """

        src_list, dst_list, ic_list, p_list = [], [], [], []
        K = self.kernel_size
        S = self.stride
        H, W = self.height, self.width
        out_h, out_w = self.out_height, self.out_width

        # After the window has been moved up/left by pad_top/pad_left, regardless of even/odd numbers, goes around K×K
        for ic in range(self.in_channels):
            for out_r_idx in range(out_h):
                for out_c_idx in range(out_w):
                    # Calculate the starting row and column indices for the kernel in the input image
                    row_start = out_r_idx * S - self.pad_top
                    col_start = out_c_idx * S - self.pad_left
                    dst = out_r_idx * out_w + out_c_idx
                    
                    # Iterate over the kernel positions
                    p = 0
                    for dr in range(0, K):    
                        for dc in range(0, K):  
                            in_r = row_start + dr
                            in_c = col_start + dc
                            # If the kernel is out of bounds, skip it
                            if self.padding == 'same':
                                if not (0 <= in_r < H and 0 <= in_c < W):
                                    p += 1
                                    continue
                            
                            if self.padding == 'valid':
                                if not (0 <= in_r < H and 0 <= in_c < W):
                                    raise RuntimeError(f"Invalid kernel region at out=({out_r_idx},{out_c_idx}), dr={dr},dc={dc}")
                            
                            # Connect node edge
                            src = in_r * W + in_c
                            src_list.append(src)
                            dst_list.append(dst)
                            ic_list.append(ic)
                            p_list.append(p)
                            p += 1

        return src_list, dst_list, ic_list, p_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Device Configuration
        device = x.device
        edge_idx = self.edge_idx.to(device) 
        ic_idx = self.ic_idx.to(device) 
        p_idx = self.p_idx.to(device) 

        # single image: (C, H, W)
        if x.dim() == 3:
            C, H, W = x.shape
            x_flat = x.permute(1, 2, 0).reshape(-1, C).to(device)
            out = self.propagate(
                edge_idx, x=x_flat,
                ic_idx=ic_idx, p_idx=p_idx,
                size=(self.num_in_nodes, self.num_out_nodes)
            )
            if self.bias is not None:
                out = out + self.bias
            out = out.view(self.out_height, self.out_width, self.out_channels)
            return out.permute(2, 0, 1)

        # batched : (B, C, H, W)
        elif x.dim() == 4:
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C).to(device)

            # edge_idx offset addition
            num_in, num_out = self.num_in_nodes, self.num_out_nodes
            src, dst = self.edge_idx
            src_batches = [src + b * num_in for b in range(B)]
            dst_batches = [dst + b * num_out for b in range(B)]
            batched_edge_idx = torch.stack([
                torch.cat(src_batches, dim=0),
                torch.cat(dst_batches, dim=0)
            ], dim=0)

            # ic_idx, p_idx copy
            batched_ic_idx = self.ic_idx.unsqueeze(0).repeat(B, 1).view(-1)
            batched_p_idx = self.p_idx.unsqueeze(0).repeat(B, 1).view(-1)

            # Message Passing
            batched_edge_idx = batched_edge_idx.to(device)
            batched_ic_idx = batched_ic_idx.to(device)
            batched_p_idx = batched_p_idx.to(device)
            out_flat = self.propagate(
                batched_edge_idx,
                x=x_flat,
                ic_idx=batched_ic_idx,
                p_idx=batched_p_idx,
                size=(B * num_in, B * num_out)
            )
            # Add bias if exists
            if self.bias is not None:
                out_flat = out_flat + self.bias.to(device)

            # Reshape
            out = out_flat.view(B, self.out_height, self.out_width, self.out_channels)
            return out.permute(0, 3, 1, 2)  # (B, out_channels, out_h, out_w)

        else:
            raise ValueError("Input must be 3D or 4D tensor")

    # Message Function(dot product)
    def message(self, x_j: torch.Tensor, ic_idx: torch.Tensor, p_idx: torch.Tensor) -> torch.Tensor:
        # x_j           : [B*E, in_channels]
        # ic_idx, p_idx : [B*E]
        # Returns: [B*E, out_channels]
        
        device = x_j.device
        x_scalar = x_j[torch.arange(x_j.size(0), device=device), ic_idx] 
        edge_weight = self.weight[ic_idx, :, p_idx]
        return edge_weight * x_scalar.unsqueeze(-1)

    # Update Function(Identity)
    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return aggr_out




if __name__ == "__main__":
    print(f"\n=============== Experiment 1 ===============")
    torch.manual_seed(10)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Verification script
    H, W, Cin, Cout, K, S = 32, 32, 3, 32, 7, 2
    B = 1000
    # Random input image
    x_img = torch.randn(B, Cin, H, W).to(device)

    # Padding type
    for padding in ['same', 'valid']:
        # Instantiate GNN layer and standard Conv2d
        gnn = Conv2dAsGNN(H, W, K, S, Cin, Cout, padding=padding, bias=True).to(device)
        if padding == 'same':
            conv = nn.Conv2d(Cin, Cout, K, stride=S, padding=K // 2, bias=True).to(device)
        else:  # valid
            conv = nn.Conv2d(Cin, Cout, K, stride=S, padding=0, bias=True).to(device)

        # Copy weights and bias from gnn to conv (preserving correlation behavior)
        with torch.no_grad():
            # Copy kernel weights
            for ic in range(Cin):
                for oc in range(Cout):
                    for p in range(K * K):
                        # map p to dr, dc in row-major order
                        dr = p // K - (K // 2)
                        dc = p % K - (K // 2)
                        conv.weight[oc, ic, dr + K // 2, dc + K // 2] = gnn.weight[ic, oc, p]
            # Copy bias
            conv.bias.copy_(gnn.bias)

        # Run both layers
        y_gnn = gnn(x_img)
        y_conv = conv(x_img)

        # Compute difference
        abs_diff = (y_gnn - y_conv).abs()
        abs_flat = abs_diff.reshape(B, -1)
        MAE_per_sample = abs_flat.mean(dim=1)

        print("Output shape (GNN)            :", y_gnn.shape)
        print("Output shape (Conv2D)         :", y_conv.shape)
        print(f"MAE                          :{MAE_per_sample.mean():.3e}")


    '''
    Conv1D
    '''
    # Sequence length, input channels, output channels, kernel size, stride
    L, Cin, Cout, K, S = 64, 3, 16, 5, 2
    B = 10000
    
    # Random input sequence
    x_seq = torch.randn(B, Cin, L).to(device)

    for padding in ['same', 'valid']:
        # GNN-based Conv1d
        gnn = Conv1dAsGNN(
            length=L, kernel_size=K, stride=S,
            in_channels=Cin, out_channels=Cout,
            padding=padding, bias=True
        ).to(device)

        # Standard Conv1d
        if padding == 'same':
            conv = nn.Conv1d(Cin, Cout, K, stride=S, padding=K // 2, bias=True).to(device)
        else:
            conv = nn.Conv1d(Cin, Cout, K, stride=S, padding=0, bias=True).to(device)

        # GNN weight → Conv1d weight Copy
        with torch.no_grad():
            # weight shape: (Cout, Cin, K)
            for ic in range(Cin):
                for oc in range(Cout):
                    for p in range(K):
                        # gnn.weight: (in_ch, out_ch, K)
                        conv.weight[oc, ic, p] = gnn.weight[ic, oc, p]
            # bias Copy
            conv.bias.copy_(gnn.bias)

        # Forward pass both layers
        y_gnn = gnn(x_seq)  # (B, Cout, L_out)
        y_conv = conv(x_seq)  # (B, Cout, L_out)

        # Difference calculation
        abs_diff = (y_gnn - y_conv).abs()
        abs_flat = abs_diff.reshape(B, -1)
        MAE_per_sample = abs_flat.mean(dim=1)

        print(f"=== padding={padding!r} ===")
        print("Output shape (GNN)            :", y_gnn.shape)
        print("Output shape (Conv1D)         :", y_conv.shape)
        print(f"MAE                          :{MAE_per_sample.mean():.3e}")

        # Input: B=2, C=3, H=W=32
        B, C, H, W = 10000, 3, 32, 32
        x_img = torch.randn(B, C, H, W)

        # GNN-based Conv: kernel_size=8, stride=8, padding='valid'
        gnn_conv = Conv2dAsGNN(
            height=H, width=W,
            kernel_size=8, stride=8,
            in_channels=3, out_channels=16,
            padding='valid', bias=True
        )

        # Standard nn.Conv2d
        conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16,
            kernel_size=8, stride=8, padding=0, bias=True
        )

        with torch.no_grad():
            for ic in range(3):
                for oc in range(16):
                    for p in range(8 * 8):
                        dr = p // 8
                        dc = p % 8
                        conv.weight[oc, ic, dr, dc] = gnn_conv.weight[ic, oc, p]
            conv.bias.copy_(gnn_conv.bias)

        # Compare outputs
        y_gnn = gnn_conv(x_img)  # (B, 16, 4, 4)
        y_conv = conv(x_img)  # (B, 16, 4, 4)

        # MAE
        abs_diff = (y_gnn - y_conv).abs()
        abs_flat = abs_diff.reshape(B, -1)
        MAE_per_sample = abs_flat.mean(dim=1)

        print("Output shape (GNN)             :", y_gnn.shape)
        print("Output shape (Conv2D)          :", y_conv.shape)
        print(f"MAE                           :{MAE_per_sample.mean():.3e}")

    
    print(f"\n=============== Experiment 2 ===============")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_iters = 10000 
    B = 1              
    Cin, Cout = 3, 32  
    H, W = 32, 32      

    # Kernel size and stride pairs
    ks_pairs = [
        (1, 1), (1, 2),
        (2, 1), (2, 2),
        (3, 1), (3, 2),
        (4, 1), (4, 2),
        (5, 1), (5, 2),
        (6, 1), (6, 2),
        (7, 1), (7, 2),
        (8, 1), (8, 2),
        (9, 1), (9, 2),
        (10, 1), (10, 2),
        (11, 1), (11, 2),
    ]

    for (K, S) in ks_pairs:
        print(f"\n=== Conv2D 비교: K={K}, S={S} ===")

        for padding in ['same', 'valid']:
            gnn_conv2d = Conv2dAsGNN(
                height=H, width=W,
                kernel_size=K, stride=S,
                in_channels=Cin, out_channels=Cout,
                padding=padding, bias=True
            ).to(device)

            if padding == 'same':
                conv2d = nn.Conv2d(Cin, Cout, K, stride=S, padding=K // 2, bias=True).to(device)
            else:  # 'valid'
                conv2d = nn.Conv2d(Cin, Cout, K, stride=S, padding=0, bias=True).to(device)

            with torch.no_grad():
                for ic in range(Cin):
                    for oc in range(Cout):
                        for p in range(K * K):
                            dr = p // K
                            dc = p % K
                            conv2d.weight[oc, ic, dr, dc] = gnn_conv2d.weight[ic, oc, p]
                conv2d.bias.copy_(gnn_conv2d.bias)

            # MAE Calculation
            total_diff = 0.0
            for _ in range(num_iters):
                # 매 반복마다 배치 크기 = 1인 랜덤 입력 생성
                x_img = torch.randn(B, Cin, H, W, device=device)
                y_gnn = gnn_conv2d(x_img)
                y_conv = conv2d(x_img)
                diff = (y_gnn - y_conv).abs().mean().item() 
                total_diff += diff

            # Input Shape 
            x_sample = torch.randn(B, Cin, H, W, device=device)
            out_shape_gnn  = tuple(gnn_conv2d(x_sample).shape)
            out_shape_conv = tuple(conv2d(x_sample).shape)
            avg_mae = total_diff / num_iters

            print(f"padding={padding!r} → Output shapes: GNN{out_shape_gnn}, Conv{out_shape_conv}")
            print(f"MAE 누적 합계: {total_diff:.6e}, 회당 평균 MAE: {avg_mae:.6e}")



    ######################## Conv1D #########################
    num_iters = 10000  
    B = 1              
    Cin, Cout = 3, 16  
    L = 64          

    # Kernel size and stride pairs
    ks_pairs = [
        (1, 1), (1, 2),
        (3, 1), (3, 2),
        (5, 1), (5, 2),
        (7, 1), (7, 2),
        (9, 1), (9, 2),
        (11, 1), (11, 2),
    ]

    for (K, S) in ks_pairs:
        print(f"\n=== Conv1D 비교: K={K}, S={S} ===")

        for padding in ['same', 'valid']:
            gnn_conv1d = Conv1dAsGNN(
                length=L,
                kernel_size=K,
                stride=S,
                in_channels=Cin,
                out_channels=Cout,
                padding=padding,
                bias=True
            ).to(device)

            if padding == 'same':
                conv1d = nn.Conv1d(Cin, Cout, K, stride=S, padding=K // 2, bias=True).to(device)
            else:  # 'valid'
                conv1d = nn.Conv1d(Cin, Cout, K, stride=S, padding=0, bias=True).to(device)

            with torch.no_grad():
                for ic in range(Cin):
                    for oc in range(Cout):
                        for p in range(K):
                            conv1d.weight[oc, ic, p] = gnn_conv1d.weight[ic, oc, p]
                conv1d.bias.copy_(gnn_conv1d.bias)

            # MAE Calculation
            total_diff = 0.0
            for _ in range(num_iters):
                x_seq = torch.randn(B, Cin, L, device=device)
                y_gnn = gnn_conv1d(x_seq)
                y_conv = conv1d(x_seq)
                diff = (y_gnn - y_conv).abs().mean().item()
                total_diff += diff

            # Input Shape
            x_sample = torch.randn(B, Cin, L, device=device)
            out_shape_gnn  = tuple(gnn_conv1d(x_sample).shape)
            out_shape_conv = tuple(conv1d(x_sample).shape)
            avg_mae = total_diff / num_iters
            print(f"padding={padding!r} → Output shapes: GNN{out_shape_gnn}, Conv{out_shape_conv}")
            print(f"MAE 누적 합계: {total_diff:.6e}, 회당 평균 MAE: {avg_mae:.6e}")