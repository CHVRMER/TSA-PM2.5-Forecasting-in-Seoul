"""Paper-aligned DCRNN building blocks adapted to the frozen Seoul task.

The existing support orientation/self-loop rule, diffusion basis and DCGRU cell
are preserved exactly. Configurable stacked encoder/decoder cells and sparse,
training-only Junggu scheduled sampling are explicit project adaptations.
This is not a reproduction of all-node traffic benchmark supervision.
"""
from __future__ import annotations
import inspect
import math
from typing import Sequence
import torch
from torch import nn

def diffusion_supports(adjacency, add_self_loops: bool = True) -> torch.Tensor:
    """Return forward/reverse, receiver-by-sender random-walk supports.

    For source-by-destination A, forward transport aggregates with P.T, where
    P = D_out^-1 A.  Thus a source-0 signal can reach node 1 when A[0, 1] > 0.
    Self-loops are added identically for all graph variants.  Empty rows remain
    zero when add_self_loops=False; they never produce NaNs.
    """
    adjacency = torch.as_tensor(adjacency, dtype=torch.float32).detach().clone()
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency must be a square [nodes, nodes] matrix.")
    if adjacency.shape[0] == 0:
        raise ValueError("At least one node is required.")
    if not torch.isfinite(adjacency).all() or (adjacency < 0).any():
        raise ValueError("Adjacency weights must be finite and nonnegative.")
    identity = torch.eye(len(adjacency), dtype=adjacency.dtype,
                         device=adjacency.device)
    supports = []
    for matrix in (adjacency, adjacency.T):
        matrix = matrix + identity if add_self_loops else matrix
        degree = matrix.sum(dim=1, keepdim=True)
        probability = matrix / degree.clamp_min(torch.finfo(matrix.dtype).tiny)
        supports.append(probability.T.contiguous())
    return torch.stack(supports)

class DiffusionLinear(nn.Module):
    """Linear projection of identity and K diffusion bases per support.

    Bases use x_1=Sx and x_k=2*S*x_(k-1)-x_(k-2), as in the DCRNN reference
    cell's Chebyshev-style implementation.  The recurrence is reset separately
    for each support, avoiding accidental cross-contamination of directions.
    """

    def __init__(self, in_features: int, out_features: int,
                 diffusion_steps: int = 2, num_supports: int = 2):
        super().__init__()
        if min(in_features, out_features, num_supports) < 1 or diffusion_steps < 0:
            raise ValueError("Invalid diffusion dimensions or diffusion steps.")
        self.diffusion_steps = int(diffusion_steps)
        self.num_supports = int(num_supports)
        width = in_features * (1 + num_supports * self.diffusion_steps)
        self.linear = nn.Linear(width, out_features)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, supports: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or supports.shape != (self.num_supports, x.shape[1], x.shape[1]):
            raise ValueError("Expected x=[batch,nodes,features] and matching supports.")
        basis = [x]
        if self.diffusion_steps:
            for support in supports.unbind(dim=0):
                previous, current = x, torch.matmul(support, x)
                basis.append(current)
                for _ in range(2, self.diffusion_steps + 1):
                    following = 2.0 * torch.matmul(support, current) - previous
                    basis.append(following)
                    previous, current = current, following
        return self.linear(torch.cat(basis, dim=-1))

class DCGRUCell(nn.Module):
    """A GRU cell whose gates and candidate use bidirectional diffusion."""

    def __init__(self, input_dim: int, hidden_dim: int, diffusion_steps: int = 2):
        super().__init__()
        self.input_dim, self.hidden_dim = int(input_dim), int(hidden_dim)
        self.gates = DiffusionLinear(input_dim + hidden_dim, 2 * hidden_dim,
                                    diffusion_steps=diffusion_steps)
        self.candidate = DiffusionLinear(input_dim + hidden_dim, hidden_dim,
                                        diffusion_steps=diffusion_steps)
        nn.init.ones_(self.gates.linear.bias)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor,
                supports: torch.Tensor) -> torch.Tensor:
        if inputs.shape[:-1] != hidden.shape[:-1]:
            raise ValueError("Input and hidden state must share batch/node dimensions.")
        reset, update = torch.sigmoid(
            self.gates(torch.cat((inputs, hidden), dim=-1), supports)
        ).chunk(2, dim=-1)
        candidate = torch.tanh(self.candidate(
            torch.cat((inputs, reset * hidden), dim=-1), supports))
        return update * hidden + (1.0 - update) * candidate


def scheduled_sampling_probability(batches_seen, decay_steps=3000.0):
    """Stable tau/(tau+exp(step/tau)), including extremely large step counts."""
    steps, tau = float(batches_seen), float(decay_steps)
    if not math.isfinite(steps) or steps < 0 or not math.isfinite(tau) or tau <= 0:
        raise ValueError('Sampling step must be finite/nonnegative and tau positive.')
    exponent = steps / tau - math.log(tau)
    if exponent >= 700:
        return 0.0
    if exponent <= -700:
        return 1.0
    return 1.0 / (1.0 + math.exp(exponent))


class DCRNNMultiHorizon(nn.Module):
    """Stacked DCGRU encoder and 24-hour autoregressive target-only decoder.

    forward(x, context) never accepts labels. forward_train may use only masked
    Junggu labels at supervised horizons to form the *next* decoder input.
    Other nodes' future concentrations are never accepted. Training-only sparse
    sampling differs from the paper's dense all-node scheduled sampling.

    Inputs: x=[B,T,N,2], context=[B,C]. Outputs: Junggu=[B,4].
    The second input channel is1 for missing and0 for observed. Decoder feedback
    is marked missing except a sampled, available Junggu training observation.
    """
    def __init__(self, adjacency, input_dim: int = 2, hidden_dim: int = 64,
                 diffusion_steps: int = 2,
                 horizons: Sequence[int] = (3,6,12,24), context_dim: int = 106,
                 dropout: float = 0.0, num_layers: int = 2,
                 use_curriculum_learning: bool = True, cl_decay_steps: float = 3000.,
                 decoder_start: str = 'zeros'):
        super().__init__()
        horizons=tuple(int(h) for h in horizons)
        if not horizons or min(horizons)<1 or tuple(sorted(set(horizons)))!=horizons:
            raise ValueError('Horizons must be positive, sorted and unique.')
        if input_dim!=2 or hidden_dim<1 or num_layers<1 or context_dim<0 or not 0<=dropout<1:
            raise ValueError('Invalid dimensions/dropout; PM2.5 plus missing channels required.')
        if decoder_start not in {'zeros','last_observation'}:
            raise ValueError('decoder_start must be zeros or last_observation.')
        scheduled_sampling_probability(0,cl_decay_steps)
        self.register_buffer('supports',diffusion_supports(adjacency,add_self_loops=True))
        self.num_nodes=int(self.supports.shape[-1]);self.horizons=horizons
        self.context_dim=int(context_dim);self.hidden_dim=int(hidden_dim);self.input_dim=int(input_dim)
        self.num_layers=int(num_layers);self.target_index=0
        self.use_curriculum_learning=bool(use_curriculum_learning)
        self.cl_decay_steps=float(cl_decay_steps);self.decoder_start=decoder_start
        self.encoder=nn.ModuleList([DCGRUCell(input_dim if layer==0 else hidden_dim,
                                             hidden_dim,diffusion_steps) for layer in range(num_layers)])
        self.decoder=nn.ModuleList([DCGRUCell(input_dim if layer==0 else hidden_dim,
                                             hidden_dim,diffusion_steps) for layer in range(num_layers)])
        self.node_projection=nn.Linear(hidden_dim,1)
        self.state_dropout=nn.Dropout(dropout)
        self.context_head=(nn.Sequential(nn.Linear(context_dim,hidden_dim),nn.ReLU(),nn.Dropout(dropout),
                                        nn.Linear(hidden_dim,len(horizons))) if context_dim else None)
        self.architecture_config={
            'implementation':'dcrnn-seoul-paper-aligned-v2', 'input_dim':input_dim,
            'hidden_dim':hidden_dim,'diffusion_steps':diffusion_steps,'horizons':list(horizons),
            'context_dim':context_dim,'dropout':dropout,'encoder_layers':num_layers,'decoder_layers':num_layers,
            'decoder_steps':max(horizons),'decoder_start':decoder_start,
            'use_curriculum_learning':bool(use_curriculum_learning),'cl_decay_steps':float(cl_decay_steps),
            'teacher_forcing':'training-only sparse available Junggu labels; never other-node targets',
            'supervised_node':0,'support_orientation':'A[source,destination]; receiver-by-sender supports',
            'support_self_loops':True,'per_support_recurrence_reset':True,
            'paper_benchmark_reproduction':False,
        }
        self.last_sampling_probability=0.0

    def sampling_probability(self,batches_seen):
        probability=scheduled_sampling_probability(batches_seen,self.cl_decay_steps)
        return probability if self.use_curriculum_learning else 0.0

    def _validate(self,x,context):
        if x.ndim!=4 or x.shape[1]<1 or x.shape[2:]!=(self.num_nodes,self.input_dim):
            raise ValueError('Expected x=[batch,history_hours,nodes,2].')
        if self.context_dim:
            if context is None or context.shape!=(x.shape[0],self.context_dim):
                raise ValueError('Context must match [batch,context_dim].')
        elif context is not None and context.shape!=(x.shape[0],0):
            raise ValueError('Model has no global context channels.')

    def _encode(self,x):
        states=[x.new_zeros((x.shape[0],self.num_nodes,self.hidden_dim)) for _ in range(self.num_layers)]
        for hour in range(x.shape[1]):
            value=x[:,hour]
            for layer,cell in enumerate(self.encoder):
                states[layer]=cell(value,states[layer],self.supports)
                value=self.state_dropout(states[layer]) if layer+1<self.num_layers else states[layer]
        return [self.state_dropout(state) for state in states]

    def _run(self,x,context,targets=None,mask=None,batches_seen=0):
        self._validate(x,context)
        states=self._encode(x)
        decoder_input=torch.zeros_like(x[:,-1]) if self.decoder_start=='zeros' else x[:,-1]
        use_targets=targets is not None and self.use_curriculum_learning
        probability=self.sampling_probability(batches_seen) if use_targets else 0.
        self.last_sampling_probability=probability
        requested={h:j for j,h in enumerate(self.horizons)};selected=[]
        for lead in range(1,max(self.horizons)+1):
            value=decoder_input
            for layer,cell in enumerate(self.decoder):
                states[layer]=cell(value,states[layer],self.supports)
                value=self.state_dropout(states[layer]) if layer+1<self.num_layers else states[layer]
            prediction=self.node_projection(states[-1])
            if lead in requested:selected.append(prediction[:,0,0])
            next_input=torch.cat((prediction,torch.ones_like(prediction)),dim=-1)
            # Match the reference's scalar-per-decoder-step sampling draw, but
            # expose only an actually observed supervised Junggu label.
            if use_targets and lead in requested and lead<max(self.horizons):
                column=requested[lead]
                available=mask[:,column]
                if bool(available.any()) and bool(torch.rand((),device=x.device)<probability):
                    next_input=next_input.clone()
                    next_input[:,0,0]=torch.where(available,targets[:,column],prediction[:,0,0])
                    next_input[:,0,1]=torch.where(available,torch.zeros_like(prediction[:,0,0]),
                                                torch.ones_like(prediction[:,0,0]))
            decoder_input=next_input
        result=torch.stack(selected,dim=1)
        if self.context_head is not None:result=result+self.context_head(context)
        return result

    def forward(self,x:torch.Tensor,context:torch.Tensor|None=None)->torch.Tensor:
        """Label-free inference in train/eval mode; no hidden target acceptance."""
        return self._run(x,context)

    def forward_train(self,x:torch.Tensor,context:torch.Tensor|None,
                      targets:torch.Tensor,mask:torch.Tensor,batches_seen:int=0)->torch.Tensor:
        if not self.training:
            raise RuntimeError('forward_train is forbidden in validation/test evaluation mode.')
        if targets.shape!=(x.shape[0],len(self.horizons)) or mask.shape!=targets.shape:
            raise ValueError('Training targets/mask must be [batch,selected_horizons], not all-node targets.')
        if targets.device!=x.device or mask.device!=x.device:
            raise ValueError('Inputs and sparse training labels must share a device.')
        mask=mask.to(dtype=torch.bool)
        if not torch.isfinite(targets[mask]).all():
            raise ValueError('Available training labels must be finite.')
        return self._run(x,context,targets,mask,batches_seen)


def test_model_contract():
    """Small CPU contract probe; the separate verifier adds numerical tests."""
    with torch.random.fork_rng():
        torch.manual_seed(1729)
        adjacency=torch.tensor([[0.,1.,0.],[0.,0.,1.],[0.,0.,0.]])
        model=DCRNNMultiHorizon(adjacency,hidden_dim=4,context_dim=3)
        x=torch.randn(2,5,3,2);x[...,1]=0
        context=torch.randn(2,3);output=model(x,context)
        assert output.shape==(2,4) and torch.isfinite(output).all()
        output.square().mean().backward()
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        model.eval()
        with torch.no_grad():assert torch.equal(model(x,context),model(x,context))
        return {'shape':list(output.shape),'finite_gradients':True,'deterministic_eval':True,
                'encoder_layers':model.num_layers,'label_free_forward':list(inspect.signature(model.forward).parameters),
                'parameters':sum(p.numel() for p in model.parameters())}
