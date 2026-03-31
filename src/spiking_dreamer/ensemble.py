#!/usr/bin/env python3
"""
Ensemble world models for uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import stack_module_state, functional_call, vmap
from typing import Dict, Tuple

from .world_model import ImprovedSpikingWorldModel


class ImprovedEnsembleSpikingWorldModel(nn.Module):
    """
    Ensemble of spiking world models for epistemic uncertainty estimation.
    Uses bootstrap sampling for diversity.
    """
    
    def __init__(
        self,
        num_models: int = 3,
        state_dim: int = 17,
        action_dim: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 2,
        spike_steps: int = 8,
        population_size: int = 8,
        num_scales: int = 3,
        dropout: float = 0.1,
        init_diversity: float = 0.05,
    ):
        super().__init__()
        self.num_models = num_models
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Create ensemble
        self.models = nn.ModuleList([
            ImprovedSpikingWorldModel(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                spike_steps=spike_steps,
                population_size=population_size,
                num_scales=num_scales,
                dropout=dropout,
            )
            for _ in range(num_models)
        ])
        
        # Add initial diversity to each ensemble member
        if init_diversity > 0:
            for i, model in enumerate(self.models):
                torch.manual_seed(42 + i * 1000)
                for param in model.parameters():
                    param.data += torch.randn_like(param.data) * init_diversity
        
        # Track epistemic uncertainty from last forward pass
        self._last_epistemic_stats = {
            'mean_epistemic': 0.0,
            'std_epistemic': 0.0,
            'max_epistemic': 0.0,
        }
        
        self._last_stats = {}
        
    def get_detailed_logs(self):
        """Return aggregated stats from last step with proper prefixes."""
        logs = {}
        for k, v in self._last_stats.items():
            if 'global' in k or 'rate' in k:
                logs[f'spike_rate/{k}'] = v
            else:
                logs[f'dynamics/{k}'] = v
        return logs

    def step(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Ensemble prediction with uncertainty."""
        all_next_states = []
        all_rewards = []
        all_logvars = []
        all_stats = []
        
        for model in self.models:
            next_state, reward, logvar, _, _, stats = model.step(state, action, deterministic=True)
            all_next_states.append(next_state)
            all_rewards.append(reward)
            all_logvars.append(logvar)
            safe_stats = {k: v.detach().cpu() for k, v in stats.items()}
            all_stats.append(safe_stats)
        
        # Stack [num_models, batch, dim]
        next_states = torch.stack(all_next_states, dim=0)
        rewards = torch.stack(all_rewards, dim=0)
        logvars = torch.stack(all_logvars, dim=0)
        
        # Ensemble mean
        mean_next_state = next_states.mean(dim=0)
        mean_reward = rewards.mean(dim=0)
        mean_logvar = logvars.mean(dim=0)
        
        # Epistemic uncertainty = ensemble variance
        epistemic_var = next_states.var(dim=0)
        mean_epistemic = epistemic_var.mean(dim=1, keepdim=True)
        
        # Aggregate stats
        agg_stats = {}
        if all_stats:
            keys = all_stats[0].keys()
            for k in keys:
                 vals = [s[k] for s in all_stats]
                 agg_stats[k] = sum(v.mean().item() for v in vals) / len(vals)
        
        self._last_stats = agg_stats
        self._last_epistemic_stats = {
            'mean_epistemic': mean_epistemic.mean().item(),
            'std_epistemic': mean_epistemic.std().item() if mean_epistemic.numel() > 1 else 0.0,
            'max_epistemic': mean_epistemic.max().item(),
        }
        
        return mean_next_state, mean_reward, mean_logvar, epistemic_var, mean_epistemic, agg_stats
    
    def compute_loss(self, batch: Dict[str, torch.Tensor], use_nll: bool = False) -> Tuple[torch.Tensor, float, float, float]:
        """Train all ensemble members with bootstrap sampling."""
        total_loss = 0.0
        total_state_mse = 0.0
        total_reward_mse = 0.0
        
        batch_size = batch['obs'].shape[0]
        
        for i, model in enumerate(self.models):
            # Bootstrap sampling: each model gets different subset
            bootstrap_idx = torch.randint(0, batch_size, (batch_size,), device=batch['obs'].device)
            
            bootstrap_batch = {
                'obs': batch['obs'][bootstrap_idx],
                'act': batch['act'][bootstrap_idx],
                'rew': batch['rew'][bootstrap_idx],
                'next_obs': batch['next_obs'][bootstrap_idx],
            }
            
            loss, state_mse, reward_mse, _ = model.compute_loss(bootstrap_batch, use_nll=use_nll)
            total_loss = total_loss + loss
            total_state_mse += state_mse
            total_reward_mse += reward_mse
        
        return total_loss / self.num_models, total_state_mse / self.num_models, total_reward_mse / self.num_models, 0.0
    
    def get_uncertainty_stats(self) -> Dict[str, float]:
        """Return epistemic uncertainty stats from last forward pass."""
        return self._last_epistemic_stats.copy()
    
    def get_codebook_stats(self) -> Dict[str, float]:
        return {}


class FastEnsembleSpikingWorldModel(nn.Module):
    """
    Optimized Ensemble using torch.vmap for parallel execution.
    Runs all models in parallel instead of a Python loop.
    Speedup: ~Num_Ensemble x faster.
    """
    
    def __init__(self, num_models=3, **kwargs):
        super().__init__()
        self.num_models = num_models
        
        # Remove ensemble-specific args if present
        if 'init_diversity' in kwargs:
            kwargs.pop('init_diversity')
            
        models = [
            ImprovedSpikingWorldModel(**kwargs)
            for _ in range(num_models)
        ]
        self.models = nn.ModuleList(models)
        
        # Stack parameters for vectorization
        self.params, self.buffers = stack_module_state(self.models)
        
        # Define the functional call for a SINGLE model
        self.base_model = self.models[0]
        
        # Track epistemic uncertainty and details
        self._last_epistemic_stats = {'mean_epistemic': 0.0, 'std_epistemic': 0.0, 'max_epistemic': 0.0}
        self._last_stats = {}
        
    def _sync_params(self):
        """Re-stack model parameters so vmap uses up-to-date weights after optimizer steps."""
        self.params, self.buffers = stack_module_state(self.models)

    def forward_parallel(self, obs, act):
        """Helper to run vmap step."""
        def call_single_model(params, buffers, o, a):
            return functional_call(self.base_model, (params, buffers), (o, a), kwargs={'deterministic': True})

        return vmap(call_single_model, in_dims=(0, 0, None, None), randomness='different')(self.params, self.buffers, obs, act)

    def step(self, state, action, deterministic=True):
        self._sync_params()
        # Run all models in parallel
        next_states, rewards, logvars, _, _, stats = self.forward_parallel(state, action)
        
        # Aggregation
        mean_next_state = next_states.mean(dim=0)
        mean_reward = rewards.mean(dim=0)
        mean_logvar = logvars.mean(dim=0)
        
        # Epistemic Uncertainty
        epistemic_var = next_states.var(dim=0) 
        mean_epistemic = epistemic_var.mean(dim=1, keepdim=True)
        
        # Update tracked stats
        with torch.no_grad():
            self._last_epistemic_stats = {
                'mean_epistemic': mean_epistemic.mean().item(),
                'std_epistemic': mean_epistemic.std().item() if mean_epistemic.numel() > 1 else 0.0,
                'max_epistemic': mean_epistemic.max().item(),
            }
            
            agg_stats = {}
            for k, v in stats.items():
                agg_stats[k] = v.float().mean().item()
            
            self._last_stats = agg_stats
        
        return mean_next_state, mean_reward, mean_logvar, epistemic_var, mean_epistemic, agg_stats

    def compute_loss(self, batch, use_nll=False):
        """Optimized parallel loss computation."""
        self._sync_params()
        batch_size = batch['obs'].shape[0]
        device = batch['obs'].device
        
        # Generate Bootstrap Indices
        indices = torch.randint(0, batch_size, (self.num_models, batch_size), device=device)
        
        # Gather Data
        obs = batch['obs'][indices]
        act = batch['act'][indices]
        rew = batch['rew'][indices]
        next_obs = batch['next_obs'][indices]
        
        def compute_single_loss(params, buffers, o, a, r, no):
            pred_next, pred_rew, logvar, _, _, _ = functional_call(
                self.base_model, (params, buffers), (o, a), kwargs={'deterministic': True}
            )
            
            pred_delta = pred_next - o
            target_delta = no - o
            
            if use_nll:
                inv_var = torch.exp(-logvar)
                state_loss = 0.5 * torch.mean((pred_delta - target_delta).pow(2) * inv_var) + 0.5 * torch.mean(logvar)
            else:
                state_loss = F.mse_loss(pred_delta, target_delta)
                
            reward_loss = F.mse_loss(pred_rew.squeeze(-1), r)
            total = state_loss + 0.5 * reward_loss
            return total, state_loss, reward_loss

        losses, state_mses, reward_mses = vmap(compute_single_loss, randomness='different')(
            self.params, self.buffers, obs, act, rew, next_obs
        )
        
        return losses.mean(), state_mses.mean(), reward_mses.mean(), 0.0

    def get_detailed_logs(self):
        """Aggregate detailed logs for the FastEnsemble."""
        logs = {}
        
        try:
             tau_mems = []
             tau_adapts = []
             delta_direct_norms = []
             
             for name, param in self.params.items():
                 if 'log_tau_mem' in name:
                     tau_mems.append(torch.exp(param).mean().item())
                 if 'log_tau_adapt' in name:
                     tau_adapts.append(torch.exp(param).mean().item())
                 if 'delta_direct.weight' in name:
                     delta_direct_norms.append(param.norm().item())
            
             if tau_mems:
                 logs['dynamics/tau_mem_mean'] = sum(tau_mems) / len(tau_mems)
             if tau_adapts:
                 logs['dynamics/tau_adapt_mean'] = sum(tau_adapts) / len(tau_adapts)
             if delta_direct_norms:
                 logs['weights/delta_direct_norm'] = sum(delta_direct_norms) / len(delta_direct_norms)
                 
        except Exception:
            pass
        
        if hasattr(self, '_last_stats'):
            for k, v in self._last_stats.items():
                if 'global' in k:
                    logs[f'spike_rate/{k}'] = v
                elif 'rate_' in k:
                    logs[f'spike_rate/{k}'] = v
                else:
                    logs[f'dynamics/{k}'] = v
            
        return logs

    def get_uncertainty_stats(self) -> Dict[str, float]:
        """Return epistemic uncertainty stats."""
        return self._last_epistemic_stats.copy()
    
    def get_codebook_stats(self) -> Dict[str, float]:
        return {}
