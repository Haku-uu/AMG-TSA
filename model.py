import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeConvolution(nn.Module):
    def forward(self, x, adjacency):
        # x: [B, C, N, T]
        if adjacency.dim() == 2:
            return torch.einsum("bcnt,nm->bcmt", x, adjacency).contiguous()
        if adjacency.dim() == 3:
            return torch.einsum("bcnt,bnm->bcmt", x, adjacency).contiguous()
        raise ValueError(f"Unsupported adjacency dimension: {adjacency.dim()}")


class PointwiseLinear(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        return self.proj(x)


class AsymptoticGraphConstructor(nn.Module):
    def __init__(self, signal_window=12):
        super().__init__()
        self.signal_window = signal_window
        self.W_adj = nn.Parameter(torch.empty(signal_window, signal_window))
        nn.init.xavier_uniform_(self.W_adj)

    def forward(self, traffic_signal):
        # traffic_signal: [B, N, T]
        if traffic_signal.size(-1) != self.signal_window:
            if traffic_signal.size(-1) > self.signal_window:
                traffic_signal = traffic_signal[..., -self.signal_window:]
            else:
                traffic_signal = F.pad(
                    traffic_signal,
                    (self.signal_window - traffic_signal.size(-1), 0),
                    mode="replicate",
                )

        x_min = traffic_signal.min(dim=-1, keepdim=True).values
        x_max = traffic_signal.max(dim=-1, keepdim=True).values
        x_norm = (traffic_signal - x_min) / (x_max - x_min).clamp_min(1e-6)
        x_norm = torch.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)
        x_hat = F.normalize(x_norm, p=2, dim=-1, eps=1e-6)

        # Eq. (4): x_i^T W_adj x_j, followed by ReLU and softmax.
        transformed = torch.einsum("bnt,td->bnd", x_hat, self.W_adj)
        similarity = torch.bmm(transformed, x_hat.transpose(1, 2))
        asymptotic_adj = F.softmax(F.relu(similarity), dim=-1)
        return asymptotic_adj


class AsymptoticStaticGraphConvolution(nn.Module):
    def __init__(self, c_in, c_out, dropout, diffusion_order=2):
        super().__init__()
        self.node_conv = NodeConvolution()
        self.diffusion_order = diffusion_order
        # X + K forward terms + K backward terms + one As-Graph term
        total_terms = 1 + 2 * diffusion_order + 1
        self.projection = PointwiseLinear(total_terms * c_in, c_out)
        self.dropout = dropout

    def _diffusion_terms(self, x, transition):
        terms = []
        x_k = x
        for _ in range(self.diffusion_order):
            x_k = self.node_conv(x_k, transition)
            terms.append(x_k)
        return terms

    def forward(self, x, forward_transition, backward_transition, asymptotic_adj):
        terms = [x]
        terms.extend(self._diffusion_terms(x, forward_transition))
        terms.extend(self._diffusion_terms(x, backward_transition))
        terms.append(self.node_conv(x, asymptotic_adj))
        h = self.projection(torch.cat(terms, dim=1))
        return F.dropout(h, self.dropout, training=self.training)


class GraphAttentionHead(nn.Module):
   
    def __init__(self, in_features, out_features, dropout, alpha=0.2):
        super().__init__()
        self.dropout = dropout
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, h, adjacency):
        # h: [B, T, N, C], adjacency: [B,N,N] or [N,N]
        Wh = torch.matmul(h, self.W)
        a_src = torch.matmul(Wh, self.a[: Wh.size(-1), :])
        a_dst = torch.matmul(Wh, self.a[Wh.size(-1) :, :])
        e = self.leaky_relu(a_src + a_dst.transpose(2, 3))

        if adjacency.dim() == 2:
            mask = adjacency.unsqueeze(0).unsqueeze(0) > 0
        elif adjacency.dim() == 3:
            mask = adjacency.unsqueeze(1) > 0
        else:
            raise ValueError(f"Unsupported adjacency dimension: {adjacency.dim()}")

        attention = e.masked_fill(~mask, -1e9)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        return torch.matmul(attention, Wh)


class AsymptoticDynamicGraphConvolution(nn.Module):
    """ADG module (Sec. 4.4) using the As-Graph as the GAT prior structure."""

    def __init__(self, c_in, c_out, dropout, alpha=0.2, num_heads=4):
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphAttentionHead(c_in, c_out, dropout, alpha) for _ in range(num_heads)]
        )

    def forward(self, x, asymptotic_adj):
        # Eq. (10): aggregate multiple graph-attention heads.
        head_outputs = torch.stack([head(x, asymptotic_adj) for head in self.heads], dim=0)
        return F.elu(head_outputs.mean(dim=0))


class SpatioTemporalFusion(nn.Module):
    """Spatio-temporal gated fusion in Eqs. (17)-(18)."""

    def __init__(self, channels):
       
    def forward(self, z_sg, z_dg):
        # [B,T,N,C]
        gate = torch.sigmoid(self.W_sg(z_sg) + self.W_dg(z_dg) + self.bias)
        return gate * z_sg + (1.0 - gate) * z_dg


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return torch.matmul(weights, value), weights


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class TrendAwareSelfAttention(nn.Module):
  

    def __init__(self, num_heads, d_model, kernel_size=3, dropout=0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        padding = (kernel_size - 1) // 2
        

    def _context_project(self, x, conv):
        # [B,N,T,C] -> [B,N,H,T,D]
        b, n, _, _ = x.shape
        projected = conv(x.permute(0, 3, 1, 2))
        return (
            projected.contiguous()
            .view(b, self.num_heads, self.d_k, n, -1)
            .permute(0, 3, 1, 4, 2)
        )

    def forward(self, query, key, value, mask=None):
        


class DilatedCausalConvolution(nn.Module):
    """Gated dilated causal convolution from Sec. 4.6 / Eq. (16)."""

    def __init__(self, channels, dilation_channels, kernel_size, dilation):
        super().__init__()
        self.filter_conv = nn.Conv2d(
            channels,
            dilation_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
        )
        self.gate_conv = nn.Conv2d(
            channels,
            dilation_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
        )

    def forward(self, x):
        return torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))


class AMGTSALayer(nn.Module):
    """One spatio-temporal layer in the AMG-TSA stack."""

    def __init__(
        self,
        residual_channels,
        dilation_channels,
        skip_channels,
        dropout,
        kernel_size,
        dilation,
        diffusion_order=2,
        adg_heads=4,
        tasatt_heads=8,
        use_asg=True,
        use_adg=True,
        use_tasatt=True,
    ):
        super().__init__()
        self.use_asg = use_asg
        self.use_adg = use_adg
        self.use_tasatt = use_tasatt

        self.dccn = DilatedCausalConvolution(
            residual_channels, dilation_channels, kernel_size, dilation
        )
        self.channel_projection = nn.Conv2d(dilation_channels, residual_channels, (1, 1))

        self.asg = AsymptoticStaticGraphConvolution(
            residual_channels, residual_channels, dropout, diffusion_order
        )
        self.adg = AsymptoticDynamicGraphConvolution(
            residual_channels, residual_channels, dropout, num_heads=adg_heads
        )
        self.fusion = SpatioTemporalFusion(residual_channels)
        self.tasatt = TrendAwareSelfAttention(
            tasatt_heads, residual_channels, kernel_size=3, dropout=0.0
        )
        self.batch_norm = nn.BatchNorm2d(residual_channels)
        self.skip_projection = nn.Conv2d(residual_channels, skip_channels, (1, 1))

    def forward(self, x, forward_transition, backward_transition, asymptotic_adj):
        temporal = self.channel_projection(self.dccn(x))

        if self.use_asg:
            z_sg = self.asg(
                temporal, forward_transition, backward_transition, asymptotic_adj
            )
        else:
            z_sg = temporal

        temporal_bt = temporal.transpose(1, 3)  # [B,T,N,C]
        if self.use_adg:
            z_dg = self.adg(temporal_bt, asymptotic_adj)
        else:
            z_dg = temporal_bt

        if self.use_asg and self.use_adg:
            fused = self.fusion(z_sg.transpose(1, 3), z_dg)
        elif self.use_asg:
            fused = z_sg.transpose(1, 3)
        else:
            fused = z_dg

        x = fused.permute(0, 3, 2, 1).contiguous()  # [B,C,N,T]
        x = self.batch_norm(x)

        if self.use_tasatt:
            tas_input = x.permute(0, 2, 3, 1).contiguous()  # [B,N,T,C]
            tas_output = self.tasatt(tas_input, tas_input, tas_input)
            x = tas_output.permute(0, 3, 1, 2).contiguous()

        skip = self.skip_projection(x)
        return x, skip


class AMGTSA(nn.Module):
    """Asymptotic Mixed Graph Convolutional with Trend-Aware Self-Attention Network."""

    def __init__(
        self,
        device,
        num_nodes,
        dropout=0.3,
        supports=None,
        in_dim=2,
        out_dim=12,
        residual_channels=64,
        dilation_channels=64,
        skip_channels=512,
        end_channels=1024,
        kernel_size=2,
        blocks=4,
        layers=2,
        signal_window=12,
        diffusion_order=2,
        adg_heads=4,
        tasatt_heads=8,
        use_asg=True,
        use_adg=True,
        use_tasatt=True,
        use_dccn=True,
        **legacy_kwargs,
    ):
        super().__init__()
        del device, num_nodes, legacy_kwargs

        self.blocks = blocks
        self.layers = layers
        self.use_dccn = use_dccn
        self.supports = supports if supports is not None else []
        if len(self.supports) == 0:
            raise ValueError(
                "AMG-TSA requires predefined transition matrices. "
                "Use adjtype='doubletransition' to provide forward/backward matrices."
            )

        self.input_projection = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))
        self.asymptotic_graph_constructor = AsymptoticGraphConstructor(signal_window)

        self.st_layers = nn.ModuleList()
        receptive_field = 1
        for _ in range(blocks):
            dilation = 1
            for _ in range(layers):
                layer_dilation = dilation if use_dccn else 1
                self.st_layers.append(
                    AMGTSALayer(
                        residual_channels=residual_channels,
                        dilation_channels=dilation_channels,
                        skip_channels=skip_channels,
                        dropout=dropout,
                        kernel_size=kernel_size if use_dccn else 1,
                        dilation=layer_dilation,
                        diffusion_order=diffusion_order,
                        adg_heads=adg_heads,
                        tasatt_heads=tasatt_heads,
                        use_asg=use_asg,
                        use_adg=use_adg,
                        use_tasatt=use_tasatt,
                    )
                )
                receptive_field += (kernel_size - 1) * dilation if use_dccn else 0
                dilation *= 2

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1, 1))
        self.receptive_field = receptive_field

    def forward(self, input_data):
        # input_data: [B, D, N, T]
        if input_data.size(3) < self.receptive_field:
            x_input = F.pad(input_data, (self.receptive_field - input_data.size(3), 0, 0, 0))
        else:
            x_input = input_data

        # As-Graph is constructed from the online traffic measurements and shared
        # across all stacked spatio-temporal layers (Sec. 4.1-4.2).
        traffic_signal = input_data[:, 0, :, -self.asymptotic_graph_constructor.signal_window :]
        asymptotic_adj = self.asymptotic_graph_constructor(traffic_signal)

        forward_transition = self.supports[0]
        backward_transition = self.supports[1] if len(self.supports) > 1 else self.supports[0].transpose(0, 1)

        x = self.input_projection(x_input)
        skip = None
        for layer in self.st_layers:
            x, layer_skip = layer(
                x, forward_transition, backward_transition, asymptotic_adj
            )
            if skip is None:
                skip = layer_skip
            else:
                skip = skip[..., -layer_skip.size(3) :] + layer_skip

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        return self.end_conv_2(x)
