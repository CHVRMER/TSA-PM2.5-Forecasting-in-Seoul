"""Graph WaveNet operations from the authors' architecture, with Seoul readout.

Primary sources:
https://arxiv.org/abs/1906.00121 (sections 3.3, 3.4, 4.2)
https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py
https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py

The gated temporal, graph, residual, skip and output operations follow the
author implementation. Four-dimensional feature maps consistently use Conv2d;
the legacy code's Conv1d calls with a 2D kernel are not valid in modern PyTorch.

Project adaptations: [B,T,N,2] PM2.5/missing input, optional current ASOS/calendar
context, 48 direct hourly output channels and Junggu-only supervision at
3/6/12/24/36/48h. The H48 project profile uses RF61, context dropout and
missing-aware temporal padding. These are named project adaptations.
This is not a reproduction of the traffic benchmark's data or objective.
"""
from __future__ import annotations

from typing import Sequence
import torch
from torch import nn
from torch.nn import functional as F


GRAPH_WAVENET_SOURCES = {
    'paper': 'https://arxiv.org/abs/1906.00121',
    'model': 'https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py',
    'normalization': 'https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py',
}


def wave_row_transition(adjacency) -> torch.Tensor:
    """D_out^-1 A, with A[source,destination]; zero rows remain zero.

    Identity is already a separate graph-convolution term. No extra self loops
    are added, matching the author's doubletransition normalization.
    """
    a = torch.as_tensor(adjacency, dtype=torch.float32).detach().clone()
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 1:
        raise ValueError('Adjacency must be a nonempty square matrix.')
    if not torch.isfinite(a).all() or (a < 0).any():
        raise ValueError('Adjacency must be finite and nonnegative.')
    total = a.sum(dim=1, keepdim=True)
    return torch.where(total > 0, a / total.clamp_min(torch.finfo(a.dtype).tiny), torch.zeros_like(a))


def wave_message(x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Authors' nconv: [B,C,source,T] @ P[source,destination]."""
    return torch.einsum('bcnt,nm->bcmt', x, support).contiguous()


class WaveGraphConvolution(nn.Module):
    """[X, XP1, XP1², XP2, XP2², ...] followed by pointwise projection."""
    def __init__(self, in_channels: int, out_channels: int, support_count: int,
                 order: int = 2, dropout: float = 0.3):
        super().__init__()
        if min(in_channels, out_channels, support_count, order) < 1 or not 0 <= dropout < 1:
            raise ValueError('Invalid graph-convolution configuration.')
        self.order = int(order)
        self.support_count = int(support_count)
        self.dropout = float(dropout)
        self.project = nn.Conv2d(in_channels * (1 + order * support_count), out_channels, (1, 1))

    def forward(self, x: torch.Tensor, supports: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(supports) != self.support_count:
            raise ValueError('Graph support count differs from construction.')
        values = [x]
        for support in supports:
            propagated = x
            for _ in range(self.order):
                propagated = wave_message(propagated, support)
                values.append(propagated)
        out = self.project(torch.cat(values, dim=1))
        return F.dropout(out, self.dropout, training=self.training)


class GraphWaveNetMultiHorizon(nn.Module):
    """Author-style direct forecast model and an explicit Seoul output adapter.

    graph_mode:
      fixed_only: two directed random-walk supports, no learned adjacency.
      fixed_plus_adaptive: those two supports and learned softmax(ReLU(E1 E2)).
      adaptive_only: learned adjacency only, no fixed graph information.

    The paper/code default dilation schedule has receptive_field=13. With a
    48-hour input and last-time readout in eval mode, only the latest13 PM hours
    affect that readout. Training BatchNorm also estimates statistics across
    historical positions, as in the author code. To cover all48 PM hours with
    the eval-time convolutional readout, use an explicitly named project profile
    with layers=4, blocks=4 (RF61), or another documented schedule. Do not label
    that enlarged receptive field as an unchanged original-paper setting.
    """
    def __init__(self, adjacency=None, *, num_nodes: int | None = None,
                 input_dim: int = 2, horizons: Sequence[int] = (3, 6, 12, 24, 36, 48),
                 output_steps: int = 48, target_index: int = 0,
                 context_dim: int = 0, graph_mode: str = 'fixed_plus_adaptive',
                 residual_channels: int = 32, dilation_channels: int = 32,
                 skip_channels: int = 256, end_channels: int = 512,
                 kernel_size: int = 2, blocks: int = 4, layers: int = 4,
                 diffusion_steps: int = 2, dropout: float = 0.3,
                 embedding_dim: int = 10, embedding_init: str = 'normal',
                 architecture_profile: str | None = 'project48',
                 context_dropout: float = 0.2,
                 padding_policy: str = 'scaled_pm_zero_missing_one'):
        super().__init__()
        if graph_mode not in {'fixed_only', 'fixed_plus_adaptive', 'adaptive_only'}:
            raise ValueError('Unknown graph_mode.')
        inferred_profile = ('paper' if (blocks, layers, kernel_size) == (4, 2, 2)
                            else 'project48' if (blocks, layers, kernel_size) == (4, 4, 2)
                            else 'custom')
        if architecture_profile is not None and architecture_profile != inferred_profile:
            raise ValueError('Named architecture profile and dilation dimensions disagree.')
        architecture_profile = inferred_profile
        if adjacency is not None:
            raw = torch.as_tensor(adjacency, dtype=torch.float32).detach().clone()
            forward = wave_row_transition(raw)
            if num_nodes is not None and int(num_nodes) != len(raw):
                raise ValueError('Adjacency and num_nodes differ.')
            num_nodes = len(raw)
        elif graph_mode != 'adaptive_only':
            raise ValueError('A fixed graph is required by this graph_mode.')
        else:
            raw = None
            forward = None
        if num_nodes is None or int(num_nodes) < 1:
            raise ValueError('Positive num_nodes is required.')
        n = int(num_nodes)
        horizons = tuple(int(h) for h in horizons)
        sizes = (input_dim, output_steps, residual_channels, dilation_channels,
                 skip_channels, end_channels, blocks, layers, diffusion_steps, embedding_dim)
        if min(sizes) < 1 or kernel_size < 2 or not 0 <= dropout < 1:
            raise ValueError('Invalid model dimensions or dropout.')
        if input_dim != 2 or not 0 <= context_dropout < 1:
            raise ValueError('H48 requires PM/missing input channels and valid context dropout.')
        if padding_policy != 'scaled_pm_zero_missing_one':
            raise ValueError('H48 padding must represent absent PM observations explicitly.')
        if not horizons or tuple(sorted(set(horizons))) != horizons or min(horizons) < 1 or max(horizons) > output_steps:
            raise ValueError('Horizons must be unique increasing output-step indices.')
        if not 0 <= int(target_index) < n or context_dim < 0:
            raise ValueError('Invalid target/context dimension.')
        self.num_nodes = n
        self.input_dim = int(input_dim)
        self.horizons = horizons
        self.output_steps = int(output_steps)
        self.target_index = int(target_index)
        self.context_dim = int(context_dim)
        self.graph_mode = graph_mode
        self.dropout = float(dropout)
        self.context_dropout = float(context_dropout)
        self.padding_policy = padding_policy
        self.has_adaptive = graph_mode != 'fixed_only'
        fixed = (torch.stack((forward, wave_row_transition(raw.T)))
                 if graph_mode != 'adaptive_only' else torch.empty(0, n, n))
        self.register_buffer('fixed_supports', fixed)
        self.register_buffer('raw_adjacency', raw if graph_mode != 'adaptive_only' else torch.empty(0, 0))
        if self.has_adaptive:
            self.nodevec1 = nn.Parameter(torch.empty(n, embedding_dim))
            self.nodevec2 = nn.Parameter(torch.empty(embedding_dim, n))
            if embedding_init == 'normal':
                # Public author code initializes with torch.randn.
                nn.init.normal_(self.nodevec1)
                nn.init.normal_(self.nodevec2)
            elif embedding_init == 'uniform':
                # Paper says uniform, without numerical bounds. [0,1) is an
                # explicit chosen bound and is not attributed to the paper.
                nn.init.uniform_(self.nodevec1, 0.0, 1.0)
                nn.init.uniform_(self.nodevec2, 0.0, 1.0)
            elif embedding_init == 'svd':
                if raw is None:
                    raise ValueError('SVD initialization requires an explicit adjacency.')
                u, s, vh = torch.linalg.svd(raw, full_matrices=False)
                rank = min(embedding_dim, n)
                with torch.no_grad():
                    self.nodevec1.zero_(); self.nodevec2.zero_()
                    self.nodevec1[:, :rank].copy_(u[:, :rank] * s[:rank].sqrt())
                    self.nodevec2[:rank].copy_(s[:rank, None].sqrt() * vh[:rank])
            else:
                raise ValueError('embedding_init must be normal, uniform or svd.')
        else:
            self.register_parameter('nodevec1', None)
            self.register_parameter('nodevec2', None)
        supports = len(fixed) + int(self.has_adaptive)
        self.start_conv = nn.Conv2d(input_dim, residual_channels, (1, 1))
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.graph_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        dilations = []
        for _ in range(blocks):
            for layer in range(layers):
                dilation = 2 ** layer
                dilations.append(dilation)
                self.filter_convs.append(nn.Conv2d(residual_channels, dilation_channels,
                                                  (1, kernel_size), dilation=(1, dilation)))
                self.gate_convs.append(nn.Conv2d(residual_channels, dilation_channels,
                                                (1, kernel_size), dilation=(1, dilation)))
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, (1, 1)))
                self.graph_convs.append(WaveGraphConvolution(
                    dilation_channels, residual_channels, supports, diffusion_steps, dropout))
                self.batch_norms.append(nn.BatchNorm2d(residual_channels))
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, (1, 1))
        self.end_conv_2 = nn.Conv2d(end_channels, output_steps, (1, 1))
        self.context_head = (nn.Sequential(
            nn.Linear(context_dim, residual_channels), nn.ReLU(), nn.Dropout(context_dropout),
            nn.Linear(residual_channels, len(horizons))) if context_dim else None)
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)
        self.architecture_config = {
            'implementation': 'gwn-seoul-h48-rf61-v1',
            'model': 'GraphWaveNet', 'num_nodes': n, 'input_dim': input_dim,
            'architecture_profile': architecture_profile,
            'output_steps': output_steps, 'horizons': list(horizons),
            'target_index': self.target_index, 'context_dim': context_dim,
            'context_head_is_project_adaptation': bool(context_dim),
            'context_dropout': float(context_dropout), 'padding_policy': padding_policy,
            'padding_scaled_pm_value': 0., 'padding_missing_indicator': 1.,
            'graph_mode': graph_mode, 'fixed_support_count': len(fixed),
            'adaptive_embedding_dim': embedding_dim if self.has_adaptive else 0,
            'adaptive_embedding_init': embedding_init if self.has_adaptive else None,
            'residual_channels': residual_channels, 'dilation_channels': dilation_channels,
            'skip_channels': skip_channels, 'end_channels': end_channels,
            'kernel_size': kernel_size, 'blocks': blocks, 'layers_per_block': layers,
            'dilations': dilations, 'receptive_field_hours': self.receptive_field,
            'receptive_field_scope': 'eval-time convolutional PM readout; training BatchNorm aggregates historical positions',
            'diffusion_steps': diffusion_steps, 'graph_dropout': dropout,
            'support_convention': 'P[source,destination], output_destination=sum_source(X_source*P)',
            'self_loops_added': False, 'graph_identity_term': True,
            'batch_normalization': True, 'direct_forecast': True,
            'benchmark_reproduction': False,
        }

    def adaptive_adjacency(self, detach: bool = True) -> torch.Tensor | None:
        if not self.has_adaptive:
            return None
        value = torch.softmax(torch.relu(self.nodevec1 @ self.nodevec2), dim=1)
        return value.detach() if detach else value

    def graph_supports(self) -> list[torch.Tensor]:
        result = list(self.fixed_supports.unbind(0))
        if self.has_adaptive:
            result.append(self.adaptive_adjacency(detach=False))
        return result

    def export_adaptive_adjacency(self):
        """Return a detached CPU NumPy matrix, or None for fixed_only.

        This matrix is a learned predictive dependency under the author's
        source-to-destination nconv convention, not a source contribution.
        """
        value = self.adaptive_adjacency(detach=True)
        return None if value is None else value.cpu().numpy().copy()

    def prepare_temporal_input(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose and left-pad PM/missing channels without altering observations.

        A 48-hour history receives13 left positions under RF61. Scaled PM is0
        and its missing indicator is1 in those positions, so padding cannot be
        mistaken for an actually observed train-mean PM concentration.
        """
        if x.ndim != 4 or x.shape[1] < 1 or tuple(x.shape[2:]) != (self.num_nodes, self.input_dim):
            raise ValueError('Expected x=[batch,history_hours,nodes,input_dim].')
        if not torch.isfinite(x).all():
            raise ValueError('Input must be normalized with finite missing-value placeholders.')
        z = x.permute(0, 3, 2, 1).contiguous()
        if z.shape[-1] < self.receptive_field:
            count = self.receptive_field - z.shape[-1]
            padding = z.new_zeros((*z.shape[:-1], count))
            padding[:, 1, :, :] = 1.
            z = torch.cat((padding, z), dim=-1)
        return z

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Return [B,output_hours,N,valid_input_positions] before Seoul selection.

        In eval mode every position uses its trailing RF61. BatchNorm training
        statistics aggregate supplied past positions; no future data is input.
        """
        z = self.prepare_temporal_input(x)
        z = self.start_conv(z)
        skip = None
        supports = self.graph_supports()
        for filt, gate, skip_conv, graph, norm in zip(
                self.filter_convs, self.gate_convs, self.skip_convs, self.graph_convs, self.batch_norms):
            residual = z
            z = torch.tanh(filt(z)) * torch.sigmoid(gate(z))
            s = skip_conv(z)
            skip = s if skip is None else s + skip[..., -s.shape[-1]:]
            z = graph(z, supports)
            z = norm(z + residual[..., -z.shape[-1]:])
        return self.end_conv_2(F.relu(self.end_conv_1(F.relu(skip))))

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        """[B,output_hours,N] at the latest observed time, without context."""
        return self.forward_sequence(x)[..., -1]

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        output = self.forward_all(x)
        leads = torch.as_tensor([h - 1 for h in self.horizons], device=output.device)
        selected = output[:, :, self.target_index].index_select(1, leads)
        if self.context_dim:
            if context is None or context.shape != (x.shape[0], self.context_dim) or not torch.isfinite(context).all():
                raise ValueError('Expected finite context=[batch,context_dim].')
            selected = selected + self.context_head(context)
        elif context is not None and context.shape != (x.shape[0], 0):
            raise ValueError('Model was constructed without global context.')
        return selected


# Friendly stable alias for notebook factories.
GraphWaveNet = GraphWaveNetMultiHorizon
