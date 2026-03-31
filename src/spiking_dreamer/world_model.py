#!/usr/bin/env python3
"""
Improved Spiking World Model with population coding and multi-scale temporal dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from .population_coding import PopulationEncoder, PopulationDecoder
from .multiscale import MultiScaleSNNBlock


class ImprovedSpikingWorldModel(nn.Module):
    """
    Spiking world model with:
    - Population coding for input/output
    - Multi-scale temporal integration
    - Hybrid spike + membrane readout
    - Residual connections
    """
   
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        spike_steps: int = 8,  
        population_size: int = 8,
        num_scales: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.spike_steps = spike_steps
        self.num_layers = num_layers
        self.num_scales = num_scales
        
        # Track detailed stats
        self._last_forward_stats = {}
        
        # OPTIMIZATION: Compile the heavy spiking forward pass
        if hasattr(torch, "compile"):
            self.forward_spiking = torch.compile(self.forward_spiking, mode="max-autotune")
        
        # Population encoder for inputs
        self.state_encoder = PopulationEncoder(state_dim, population_size)
        self.action_encoder = PopulationEncoder(action_dim, population_size)
        
        encoded_dim = (state_dim + action_dim) * population_size

        # Input projection — gradual compression for high-dim inputs (e.g. Humanoid 3144 → 1024 → 512)
        # Single-step 3144 → 256 is a 12:1 bottleneck that loses joint coupling information.
        self.use_pre_proj = encoded_dim > hidden_dim * 2
        if self.use_pre_proj:
            pre_dim = hidden_dim * 2
            self.fc_pre = nn.Linear(encoded_dim, pre_dim)
            self.ln_pre = nn.LayerNorm(pre_dim)
            self.fc_in = nn.Linear(pre_dim, hidden_dim)
        else:
            self.fc_pre = None
            self.fc_in = nn.Linear(encoded_dim, hidden_dim)
        self.ln_in = nn.LayerNorm(hidden_dim)
        
        # Multi-scale spiking layers
        self.snn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.snn_layers.append(
                MultiScaleSNNBlock(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    num_scales=num_scales,
                    tau_range=(1.5, 8.0),
                )
            )
        
        # Residual projections (if dimensions match, identity)
        self.residuals = nn.ModuleList([
            nn.Identity() for _ in range(num_layers)
        ])
        
        # Readout combines spike rate AND membrane potential
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 2x for spike + membrane
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Population decoder for state output
        self.delta_population = nn.Linear(hidden_dim, state_dim * population_size)
        self.delta_decoder = PopulationDecoder(state_dim, population_size)
        
        # Direct delta prediction (for residual)
        self.delta_direct = nn.Linear(hidden_dim, state_dim)
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        
        # Aleatoric uncertainty (for compatibility)
        self.logvar_head = nn.Linear(hidden_dim, state_dim)
        
        # Initialize outputs small
        nn.init.zeros_(self.delta_direct.bias)
        nn.init.normal_(self.delta_direct.weight, std=0.01)
        
        self.dropout = nn.Dropout(dropout)
    
    def _init_states(self, batch_size: int, device: torch.device) -> Tuple[List, List]:
        """Initialize all layer states."""
        all_states = []
        all_prev_spikes = []
        
        features_per_scale = self.hidden_dim // self.num_scales
        
        for _ in range(self.num_layers):
            layer_states = []
            layer_prev_spikes = []
            
            for _ in range(self.num_scales):
                v = torch.zeros(batch_size, features_per_scale, device=device)
                adapt = torch.zeros(batch_size, features_per_scale, device=device)
                spike = torch.zeros(batch_size, features_per_scale, device=device)
                layer_states.append((v, adapt, spike))
                layer_prev_spikes.append(torch.zeros(batch_size, features_per_scale, device=device))
            
            all_states.append(layer_states)
            all_prev_spikes.append(layer_prev_spikes)
        
        return all_states, all_prev_spikes
    
    def forward_spiking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Multi-step spiking forward pass.
        
        Returns both spike rate and membrane readout.
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Encode input with population coding
        state_encoded = self.state_encoder(x[:, :self.state_dim])
        action_encoded = self.action_encoder(x[:, self.state_dim:])
        encoded = torch.cat([state_encoded, action_encoded], dim=-1)
        
        # Project to hidden — gradual compression if pre_proj exists
        if self.use_pre_proj:
            encoded = F.silu(self.ln_pre(self.fc_pre(encoded)))
        h_input = self.ln_in(self.fc_in(encoded))
        
        # Initialize states
        all_states, all_prev_spikes = self._init_states(batch_size, device)
        
        # Accumulators for rate coding and membrane readout
        spike_acc = torch.zeros(batch_size, self.hidden_dim, device=device)
        membrane_acc = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # Stats accumulators (sum over time, average later)
        stats_acc = {
            'rate_fast': 0.0, 'rate_med': 0.0, 'rate_slow': 0.0,
            'layer_0_rate': 0.0, 'layer_1_rate': 0.0
        }
        
        # Multi-timestep simulation
        for t in range(self.spike_steps):
            layer_input = h_input
            
            for layer_idx, snn_layer in enumerate(self.snn_layers):
                # Get previous states
                states = all_states[layer_idx]
                prev_spikes = all_prev_spikes[layer_idx]
                
                # Forward through multi-scale layer
                spike_out, membrane_out, new_states, new_prev_spikes, layer_stats = snn_layer(
                    layer_input, states, prev_spikes
                )
                
                # Residual connection
                spike_out = spike_out + self.residuals[layer_idx](layer_input)
                
                # Update states
                all_states[layer_idx] = new_states
                all_prev_spikes[layer_idx] = new_prev_spikes
                
                for k, v in layer_stats.items():
                    stats_acc[k] = stats_acc.get(k, 0.0) + v
                
                # Per-layer rates
                layer_raw_mean = 0.0
                lay_count = 0
                if 'rate_fast' in layer_stats: layer_raw_mean += layer_stats['rate_fast']; lay_count += 1
                if 'rate_med' in layer_stats: layer_raw_mean += layer_stats['rate_med']; lay_count += 1
                if 'rate_slow' in layer_stats: layer_raw_mean += layer_stats['rate_slow']; lay_count += 1
                
                if lay_count > 0:
                    raw_rate = layer_raw_mean / lay_count
                    stats_acc[f'layer_{layer_idx}_rate'] = stats_acc.get(f'layer_{layer_idx}_rate', 0.0) + raw_rate
                    stats_acc['total_raw_rate'] = stats_acc.get('total_raw_rate', 0.0) + raw_rate
                
                # Input for next layer
                layer_input = spike_out
            
            # Accumulate final layer outputs
            spike_acc = spike_acc + spike_out
            membrane_acc = membrane_acc + membrane_out
        
        # Average over time
        spike_rate = spike_acc / self.spike_steps
        membrane_avg = membrane_acc / self.spike_steps
        
        # Normalize and store stats
        denom = self.spike_steps * self.num_layers
        final_stats = {}
        for k, v in stats_acc.items():
            if 'layer_' in k:
                final_stats[k] = (v / self.spike_steps).detach().clone()
            else:
                final_stats[k] = (v / denom).detach().clone()
                
        # Global fire rate
        if 'total_raw_rate' in stats_acc:
             final_stats['global'] = (stats_acc['total_raw_rate'] / denom).detach().clone()
             final_stats['raw_spikes_mean'] = final_stats['global']
        else:
             final_stats['global'] = torch.tensor(0.0, device=device)
        
        return spike_rate, membrane_avg, final_stats
    
    def step(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Predict next state and reward.
        Returns: next_state, reward, logvar, epi_var, mean_epi, stats
        """
        x = torch.cat([state, action], dim=-1)
        
        # Get spiking representation (both spike rate and membrane)
        spike_rate, membrane_avg, stats = self.forward_spiking(x)
        
        # Combine spike and membrane for richer representation
        combined = torch.cat([spike_rate, membrane_avg], dim=-1)
        combined = self.dropout(combined)
        
        # Readout
        hidden = self.readout(combined)
        
        # Predict delta using BOTH population decoding AND direct prediction
        delta_pop = self.delta_population(hidden)
        delta_decoded = self.delta_decoder(delta_pop)
        delta_direct = self.delta_direct(hidden)
        
        # Combine (learned weighting could be added)
        delta = 0.7 * delta_decoded + 0.3 * delta_direct
        
        # Predict reward
        reward = self.reward_head(hidden)
        
        # Aleatoric uncertainty
        logvar = self.logvar_head(hidden)
        logvar = torch.clamp(logvar, min=-10.0, max=2.0)
        
        # Next state
        if deterministic:
            next_state = state + delta
        else:
            std = torch.exp(0.5 * logvar)
            next_state = state + delta + torch.randn_like(std) * std * 0.1
        
        return next_state, reward, logvar, torch.zeros_like(next_state), torch.tensor(0.0, device=state.device), stats

    def forward(self, state, action, deterministic=True):
        """Alias for step to support torch.func.functional_call"""
        return self.step(state, action, deterministic)
    
    def compute_loss(
        self, 
        batch: Dict[str, torch.Tensor],
        use_nll: bool = False,
    ) -> Tuple[torch.Tensor, float, float, float]:
        """
        Compute training loss.
        
        Args:
            batch: Dictionary with obs, act, rew, next_obs
            use_nll: If True, use negative log-likelihood with learned variance
                    If False, use simple MSE (more stable early in training)
        """
        obs, act, rew, next_obs = batch['obs'], batch['act'], batch['rew'], batch['next_obs']
        
        next_state, reward, logvar, _, _, _ = self.step(obs, act, deterministic=True)
        
        # Target delta
        target_delta = next_obs - obs
        pred_delta = next_state - obs
        
        if use_nll:
            inv_var = torch.exp(-logvar)
            mse_term = 0.5 * torch.mean((pred_delta - target_delta).pow(2) * inv_var)
            var_term = 0.5 * torch.mean(logvar)
            state_loss = mse_term + var_term
        else:
            state_loss = F.mse_loss(pred_delta, target_delta)
        
        # Reward loss
        reward_loss = F.mse_loss(reward.squeeze(-1), rew)
        
        # Raw MSE for monitoring
        raw_mse = F.mse_loss(pred_delta, target_delta).item()
        
        # Total loss
        total_loss = state_loss + 0.5 * reward_loss
        
        return total_loss, raw_mse, reward_loss.item(), 0.0

    def get_detailed_logs(self):
        """Extract detailed SNN dynamics for logging."""
        logs = {}
        
        taus_mem = []
        taus_adapt = []
        sigmas = []
        
        sigmas.append(torch.exp(self.state_encoder.log_sigma).detach().cpu())
        
        for layer in self.snn_layers:
            for sub_layer in layer.layers:
                taus_mem.append(torch.exp(sub_layer.log_tau_mem).detach().cpu())
                taus_adapt.append(torch.exp(sub_layer.log_tau_adapt).detach().cpu())
        
        if len(taus_mem) > 0:
            logs['dynamics/tau_mem_mean'] = torch.cat(taus_mem).mean().item()
            logs['dynamics/tau_adapt_mean'] = torch.cat(taus_adapt).mean().item()
        
        if len(sigmas) > 0:
            logs['dynamics/pop_sigma_mean'] = torch.cat(sigmas).mean().item()

        with torch.no_grad():
            logs['weights/delta_direct_norm'] = self.delta_direct.weight.norm().item()
            logs['weights/delta_pop_norm'] = self.delta_population.weight.norm().item()
            
        for k, v in self._last_forward_stats.items():
            if 'global' in k:
                logs[f'spike_rate/{k}'] = v
            elif 'rate_' in k:
                logs[f'spike_rate/{k}'] = v
            else:
                logs[f'dynamics/{k}'] = v
                
        return logs
