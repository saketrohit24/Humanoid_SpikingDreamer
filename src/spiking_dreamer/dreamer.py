#!/usr/bin/env python3
"""
Enhanced Dreamer for model-based imagination with uncertainty filtering.
"""

import torch
from typing import Dict, Optional

from .ensemble import FastEnsembleSpikingWorldModel


class EnhancedDreamer:
    """
    Handles dreaming with uncertainty-based filtering.
    
    Features:
    1. Dream buffer cap relative to real data (prevents flooding)
    2. Proper uncertainty filtering with MOPO penalty
    3. Termination reason tracking
    4. Adaptive threshold based on actual acceptance rates
    """
    
    def __init__(self, world_model, actor, replay_buffer, config, device):
        self.world_model = world_model
        self.actor = actor
        self.replay_buffer = replay_buffer
        self.config = config
        self.device = device
        
        self.epistemic_threshold = config.get("epistemic_threshold", 0.5)
        self.adaptive_threshold = config.get("adaptive_threshold", True)
        self.target_acceptance_rate = 0.6
        
        # Dream buffer cap (relative to real buffer)
        self.max_dream_ratio = config.get("max_dream_ratio", 10.0)
        
        # Track termination reasons
        self.last_termination_reason = "none"
        self.last_mean_uncertainty = 0.0
        
        # MOPO Penalty
        self.penalty_coef = config.get("uncertainty_penalty", 1.0)

        # Cumulative stats
        self.stats = {
            'total_generated': 0,
            'total_accepted': 0,
        }
        
    def dream_and_augment(self, batch_size, horizon, target_buffer=None, exploration_noise=0.1):
        if target_buffer is None:
            target_buffer = self.replay_buffer
        
        # Check dream buffer cap
        if hasattr(target_buffer, 'size') and hasattr(self.replay_buffer, 'real_count'):
            real_count = self.replay_buffer.real_count
            max_dreams = int(real_count * self.max_dream_ratio)
            current_dreams = target_buffer.size if target_buffer != self.replay_buffer else target_buffer.dream_count
            
            if current_dreams >= max_dreams:
                self.last_termination_reason = "buffer_cap"
                return {
                    'dreams_added': 0,
                    'acceptance_rate': 0.0,
                    'epistemic_threshold': self.epistemic_threshold,
                    'termination_reason': 'buffer_cap',
                    'mean_uncertainty': 0.0,
                }
            
        # Sample states
        if hasattr(self.replay_buffer, 'sample_states_for_dreaming'):
            obs = self.replay_buffer.sample_states_for_dreaming(batch_size, self.device)
        else:
            obs, _, _, _, _ = self.replay_buffer.sample(batch_size)
            
        curr_obs = obs
        dreams_added = 0
        total_generated = 0
        total_uncertainty = 0.0
        uncertainty_count = 0
        
        for t in range(horizon):
            with torch.no_grad():
                # Actor action - denormalize obs first (actor trained on raw env obs)
                if hasattr(self.replay_buffer, 'denormalize_obs'):
                    actor_obs = self.replay_buffer.denormalize_obs(curr_obs)
                else:
                    actor_obs = curr_obs
                action = self.actor(actor_obs)

                # Noise
                noise = torch.randn_like(action) * exploration_noise
                action = (action + noise).clamp(-1.0, 1.0)

                # World model prediction (world model expects normalized obs)
                step_results = self.world_model.step(curr_obs, action)
                if len(step_results) == 6:
                    next_obs, reward, _, _, mean_epistemic, _ = step_results
                else:
                    next_obs, reward, _, _, mean_epistemic = step_results

                # Track uncertainty stats
                uncertainty = mean_epistemic.squeeze()
                if uncertainty.dim() == 0:
                    uncertainty = uncertainty.unsqueeze(0)
                total_uncertainty += uncertainty.mean().item()
                uncertainty_count += 1

                # Fixed threshold - no aggressive horizon decay
                step_threshold = self.epistemic_threshold
                mask = uncertainty < step_threshold
                
                valid_idx = torch.where(mask)[0]
                
                if len(valid_idx) > 0:
                    # MOPO: Penalize reward
                    unc_penalty = uncertainty[valid_idx].unsqueeze(-1)
                    raw_rew = reward[valid_idx]
                    penalized_rew = raw_rew - (self.penalty_coef * unc_penalty)

                    # Prepare for buffer
                    if hasattr(self.replay_buffer, 'denormalize_obs'):
                        unnorm_obs = self.replay_buffer.denormalize_obs(curr_obs[valid_idx])
                        unnorm_next = self.replay_buffer.denormalize_obs(next_obs[valid_idx])
                    else:
                        unnorm_obs = curr_obs[valid_idx]
                        unnorm_next = next_obs[valid_idx]

                    b_act = action[valid_idx]
                    b_done = torch.zeros((len(valid_idx), 1), device=self.device)
                    
                    if hasattr(target_buffer, 'add_batch'):
                        target_buffer.add_batch(unnorm_obs, b_act, unnorm_next, penalized_rew, b_done, is_dream=True)
                    else:
                        b_obs_np = unnorm_obs.cpu().numpy()
                        b_next_np = unnorm_next.cpu().numpy()
                        b_act_np = b_act.cpu().numpy()
                        b_rew_np = penalized_rew.cpu().numpy()
                        b_done_np = b_done.cpu().numpy()
                        
                        target_buffer.add_batch(b_obs_np, b_act_np, b_next_np, b_rew_np, b_done_np, is_dream=True)
                            
                    dreams_added += len(valid_idx)
                
                total_generated += batch_size
                curr_obs = next_obs
                
        acceptance_rate = dreams_added / (total_generated + 1e-8)
        mean_uncertainty = total_uncertainty / max(1, uncertainty_count)
        
        # Determine termination reason
        if acceptance_rate < 0.1:
            termination_reason = "uncertainty"
        else:
            termination_reason = "horizon"
        
        self.last_termination_reason = termination_reason
        self.last_mean_uncertainty = mean_uncertainty
        
        # Update cumulative stats
        self.stats['total_generated'] += total_generated
        self.stats['total_accepted'] += dreams_added
        
        # Adaptive threshold update
        if self.adaptive_threshold and acceptance_rate > 0:
            self.adaptive_threshold_update(acceptance_rate)
        
        return {
            'dreams_added': dreams_added,
            'acceptance_rate': acceptance_rate,
            'epistemic_threshold': self.epistemic_threshold,
            'termination_reason': termination_reason,
            'mean_uncertainty': mean_uncertainty,
            'final_valid_ratio': dreams_added / max(1, batch_size * horizon),
        }

    def adaptive_threshold_update(self, acceptance_rate):
        if not self.adaptive_threshold:
            return
        
        if acceptance_rate < self.target_acceptance_rate - 0.1:
            # Too few accepted - increase threshold
            self.epistemic_threshold = min(self.epistemic_threshold * 1.05, 2.0)
        elif acceptance_rate > self.target_acceptance_rate + 0.1:
            # Too many accepted - decrease threshold
            self.epistemic_threshold = max(self.epistemic_threshold * 0.95, 0.01)


def create_improved_spiking_world_model(config: dict, device: torch.device):
    """Create the improved spiking world model (using FastEnsemble)."""
    return FastEnsembleSpikingWorldModel(
        num_models=config.get("num_ensemble", 3),
        state_dim=config.get("obs_dim", 17),
        action_dim=config.get("act_dim", 6),
        hidden_dim=config.get("wm_hidden_dim", 256),
        num_layers=config.get("wm_num_layers", 2),
        spike_steps=config.get("wm_spike_steps", 8),
        population_size=config.get("population_size", 8),
        num_scales=config.get("num_scales", 3),
        dropout=config.get("wm_dropout", 0.1),
        init_diversity=config.get("init_diversity", 0.02),
    ).to(device)
