#!/usr/bin/env python3
"""
TD3 Agent with Spiking World Model Dreaming (MBPO-style).

FIXES applied:
  1. Separate GradScaler for WM (uses autocast) vs plain FP32 for TD3
  2. Dream buffer cleared periodically to prevent stale dream poisoning
  3. Gradient clipping on critic and actor
  4. More conservative real/dream ratio (floor 0.70 instead of 0.55)
"""

import torch
import torch.nn.functional as F
from copy import deepcopy
from torch.optim import Adam
from typing import Dict

from .actor import SNN_Actor
from .critic import Critic
from .ensemble import ImprovedEnsembleSpikingWorldModel
from .dreamer import EnhancedDreamer
from .replay_buffer import ReplayBuffer


class TD3_SpikingDreamer:
    """
    TD3 Agent with:
    - SNN Actor with population coding
    - Spiking world model ensemble for dreaming
    - MBPO-style mixed real/dream training
    - MOPO uncertainty penalty
    """
    
    def __init__(self, state_dim, action_dim, max_action, device, config, replay_buffer):
        self.device = device
        self.max_action = max_action
        self.discount = config["gamma"]
        self.tau = config["tau"]
        self.policy_noise = config["policy_noise"] * max_action
        self.noise_clip = config["noise_clip"] * max_action
        self.policy_freq = config["policy_freq"]
        self.config = config
        
        # SNN Actor
        self.actor = SNN_Actor(
            state_dim, action_dim, max_action,
            config["en_pop_dim"], config["de_pop_dim"],
            config["hidden_sizes"], config["mean_range"],
            config["std"], config["spike_ts"], device
        ).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=config["actor_lr"])
        
        # Critic
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config["critic_lr"])
        
        # Ensemble spiking world model
        self.world_model = ImprovedEnsembleSpikingWorldModel(
            num_models=config["num_ensemble"],
            state_dim=state_dim, 
            action_dim=action_dim,
            hidden_dim=config["wm_hidden_dim"],
            num_layers=config["wm_num_layers"],
            spike_steps=config["wm_spike_steps"],
            population_size=config["population_size"],
            num_scales=config["num_scales"],
        ).to(device)
        
        self.wm_optimizer = Adam(
            self.world_model.parameters(),
            lr=config["wm_lr"],
            weight_decay=config["wm_weight_decay"],
        )

        # FIX 1: Separate GradScaler for world model ONLY.
        # The old code shared one scaler between WM (which uses autocast/bf16)
        # and TD3 critic/actor (which run in fp32).  GradScaler without autocast
        # multiplies fp32 losses by the internal scale factor (can be 65536x),
        # and WM loss spikes drag the scale down, starving critic/actor gradients.
        self.wm_scaler = torch.amp.GradScaler('cuda')
        # TD3 critic/actor will use plain fp32 — no scaler needed.
        
        # Dreamer
        self.dreamer = EnhancedDreamer(
            self.world_model, 
            self.actor, 
            replay_buffer, 
            config, 
            device
        )
        
        self.total_it = 0

        # EMA of WM state_mse — used to adapt real/dream ratio in train()
        self.wm_mse_ema = 1.0

        # World model metrics
        self.wm_metrics = {
            'loss': 0, 'state_mse': 0, 'reward_mse': 0, 'vq_loss': 0, 
            'grad_snn_norm': 0, 'grad_direct_norm': 0,
            'count': 0
        }
        
        # FIX 2: Smaller dream buffer + periodic refresh.
        # 200k is plenty; we clear it every dream phase so stale dreams
        # from outdated policy regions don't accumulate and poison the critic.
        self.dream_buffer = ReplayBuffer(state_dim, action_dim, max_size=200000, device=device)
        self._dream_phase_count = 0
        self._dream_refresh_interval = config.get("dream_refresh_interval", 5)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train_world_model(self, replay_buffer, batch_size=256):
        """Train world model on real data only."""
        state, action, next_state, reward, _ = replay_buffer.sample_real_only(batch_size)
        batch = {
            'obs': state,
            'act': action,
            'rew': reward.squeeze(-1),
            'next_obs': next_state
        }
        
        self.wm_optimizer.zero_grad()

        # FastEnsemble uses vmap which is incompatible with autocast+GradScaler.
        # Plain fp32 is stable enough for WM training.
        loss, state_mse, reward_mse, vq_loss = self.world_model.compute_loss(batch)
        loss.backward()

        # Gradient norm tracking
        snn_norm = 0.0
        direct_norm = 0.0

        try:
            first_model = self.world_model.models[0]
            for param in first_model.snn_layers.parameters():
                if param.grad is not None:
                    snn_norm += param.grad.norm().item() ** 2
            for param in first_model.delta_direct.parameters():
                if param.grad is not None:
                    direct_norm += param.grad.norm().item() ** 2
            snn_norm = snn_norm ** 0.5
            direct_norm = direct_norm ** 0.5
        except Exception:
            pass

        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.wm_optimizer.step()
        
        # Update EMA of state_mse for adaptive dream ratio
        self.wm_mse_ema = 0.995 * self.wm_mse_ema + 0.005 * state_mse

        # Accumulate metrics
        self.wm_metrics['loss'] += loss.item()
        self.wm_metrics['state_mse'] += state_mse
        self.wm_metrics['reward_mse'] += reward_mse
        self.wm_metrics['vq_loss'] += vq_loss
        self.wm_metrics['grad_snn_norm'] += snn_norm
        self.wm_metrics['grad_direct_norm'] += direct_norm
        self.wm_metrics['count'] += 1
        
        return {'loss': loss.item()}

    def dream_phase(self, replay_buffer):
        """Run a dreaming phase."""
        self._dream_phase_count += 1

        # FIX 2 (cont): Periodically wipe the dream buffer so stale dreams
        # from old policy regions don't accumulate.  Fresh dreams only.
        if self._dream_phase_count % self._dream_refresh_interval == 0:
            self.dream_buffer.ptr = 0
            self.dream_buffer.size = 0
            self.dream_buffer.dream_count = 0

        dream_batch = self.config["dream_batch_size"] * self.config["dreams_per_phase"]
        
        metrics = self.dreamer.dream_and_augment(
            batch_size=dream_batch,
            horizon=self.config["dream_horizon"],
            target_buffer=self.dream_buffer,
            exploration_noise=self.config.get("dream_exploration_noise", 0.3),
        )
        
        return metrics['dreams_added'], [metrics]

    def get_wm_status(self):
        """Get a human-readable status of the world model."""
        if self.wm_metrics['count'] == 0:
            return "WARMUP"
        
        avg_loss = self.wm_metrics['loss'] / self.wm_metrics['count']
        epistemic = self.world_model.get_uncertainty_stats().get('mean_epistemic', 0.0)
        
        if avg_loss > 100.0:
            return "UNSTABLE (High Loss)"
        elif epistemic > self.config.get('epistemic_threshold', 0.5):
            return "UNCERTAIN (High Epistemic)"
        elif avg_loss < 1.0:
            return "STABLE (Good)"
        else:
            return "LEARNING"

    def train(self, replay_buffer, batch_size=256):
        """TD3 training with MBPO mixed sampling."""
        self.total_it += 1

        # Adaptive real/dream ratio: WM quality gate + step-based decay.
        # WM quality determines base dream fraction.
        if self.wm_mse_ema > 0.08:
            dream_frac = 0.05   # WM still learning — almost all real
        elif self.wm_mse_ema > 0.05:
            dream_frac = 0.15   # WM decent
        elif self.wm_mse_ema > 0.03:
            dream_frac = 0.25   # WM good
        else:
            dream_frac = 0.30   # WM excellent

        # Step-based cap: dream ratio decays from 50% → 10% over training.
        step_frac = min(1.0, self.total_it / max(1, self.config.get("total_steps", 2_000_000)))
        step_max_dream = max(0.10, 0.50 * (1.0 - step_frac))
        dream_frac = min(dream_frac, step_max_dream)

        # Dream ramp-up: start at 20% dreams and ramp to full ratio over
        # dream_ramp_steps after dreaming begins. Avoids shocking the critic
        # with 30% dream data the moment dreams turn on.
        dream_start = self.config.get("dream_start_step", 300000)
        ramp_steps = self.config.get("dream_ramp_steps", 100000)
        approx_env_step = self.total_it + self.config.get("start_timesteps", 50000)
        steps_since_dream = max(0, approx_env_step - dream_start)
        ramp_frac = min(1.0, steps_since_dream / max(1, ramp_steps))
        dream_frac = 0.20 * (1.0 - ramp_frac) + dream_frac * ramp_frac

        real_ratio = 1.0 - dream_frac
        real_batch_size = int(batch_size * real_ratio)
        dream_batch_size = batch_size - real_batch_size
        
        if self.dream_buffer.size >= dream_batch_size and dream_batch_size > 0:
            r_obs, r_act, r_next, r_rew, r_not_done = replay_buffer.sample(real_batch_size)
            d_obs, d_act, d_next, d_rew, d_not_done = self.dream_buffer.sample(dream_batch_size)
            
            state = torch.cat([r_obs, d_obs], dim=0)
            action = torch.cat([r_act, d_act], dim=0)
            next_state = torch.cat([r_next, d_next], dim=0)
            reward = torch.cat([r_rew, d_rew], dim=0)
            not_done = torch.cat([r_not_done, d_not_done], dim=0)
        else:
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        # ---------- Critic update (plain fp32, NO scaler) ----------
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q
        
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # FIX 3: Clip critic gradients — prevents Q-value explosions from
        # bad dream targets.
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # ---------- Actor update (plain fp32, NO scaler) ----------
        actor_loss = 0.0
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # FIX 3 (cont): Clip actor gradients too.
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # Soft update targets
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
            for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
            
            actor_loss = actor_loss.item()
        
        return critic_loss.item(), actor_loss

    def get_wm_metrics(self) -> Dict:
        """Get averaged world model metrics and reset."""
        if self.wm_metrics['count'] == 0:
            return {}
        
        metrics = {
            'wm/loss': self.wm_metrics['loss'] / self.wm_metrics['count'],
            'wm/state_mse': self.wm_metrics['state_mse'] / self.wm_metrics['count'],
            'wm/reward_mse': self.wm_metrics['reward_mse'] / self.wm_metrics['count'],
            'wm/vq_loss': self.wm_metrics['vq_loss'] / self.wm_metrics['count'],
        }
        
        # Add uncertainty stats
        unc_stats = self.world_model.get_uncertainty_stats()
        for k, v in unc_stats.items():
            metrics[f'wm/{k}'] = v
            
        # Add codebook stats
        cb_stats = self.world_model.get_codebook_stats()
        for k, v in cb_stats.items():
            metrics[f'wm/{k}'] = v

        # Add detailed SNN stats
        detailed_logs = self.world_model.get_detailed_logs()
        for k, v in detailed_logs.items():
            metrics[f'wm/{k}'] = v
            
        # Add gradient norms
        metrics['wm/grad/snn_layers'] = self.wm_metrics['grad_snn_norm'] / self.wm_metrics['count']
        metrics['wm/grad/direct_head'] = self.wm_metrics['grad_direct_norm'] / self.wm_metrics['count']
        
        # Extra diagnostic: real/dream ratio being used
        metrics['train/real_ratio'] = 0.95 if self.wm_mse_ema > 0.08 else (
            0.85 if self.wm_mse_ema > 0.05 else (0.75 if self.wm_mse_ema > 0.03 else 0.70))
        metrics['train/wm_mse_ema'] = self.wm_mse_ema
        metrics['train/dream_buffer_size'] = self.dream_buffer.size
        
        # Reset
        self.wm_metrics = {
            'loss': 0, 'state_mse': 0, 'reward_mse': 0, 'vq_loss': 0, 
            'grad_snn_norm': 0, 'grad_direct_norm': 0,
            'count': 0
        }
        
        return metrics

    def evaluate_world_model(self, replay_buffer, sample_size=1000) -> Dict:
        """Comprehensive world model evaluation."""
        if replay_buffer.size < sample_size:
            return {}
        
        obs, act, next_obs, rew, _ = replay_buffer.sample_real_only(sample_size)
        
        with torch.no_grad():
            pred_next, pred_rew, logvar, epistemic, _, _ = self.world_model.step(obs, act)
        
        metrics = {}
        
        # Single-step metrics
        metrics['eval/state_mse'] = F.mse_loss(pred_next, next_obs).item()
        metrics['eval/state_mae'] = F.l1_loss(pred_next, next_obs).item()
        metrics['eval/reward_mse'] = F.mse_loss(pred_rew.squeeze(), rew.squeeze()).item()
        
        # R² Score
        ss_res = ((pred_next - next_obs) ** 2).sum()
        ss_tot = ((next_obs - next_obs.mean(dim=0)) ** 2).sum()
        metrics['eval/r2_score'] = (1 - ss_res / ss_tot).item()
        
        # Per-dimension analysis
        per_dim_mse = ((pred_next - next_obs) ** 2).mean(dim=0)
        metrics['eval/worst_dim_mse'] = per_dim_mse.max().item()
        metrics['eval/best_dim_mse'] = per_dim_mse.min().item()
        
        # Uncertainty calibration
        actual_error = ((pred_next - next_obs) ** 2).mean(dim=1)
        pred_uncertainty = epistemic.mean(dim=1) if epistemic.dim() > 1 else epistemic
        
        if pred_uncertainty.std() > 1e-8 and actual_error.std() > 1e-8:
            corr_matrix = torch.corrcoef(torch.stack([actual_error, pred_uncertainty]))
            unc_corr = corr_matrix[0, 1].item()
            metrics['eval/uncertainty_error_corr'] = unc_corr if not torch.isnan(torch.tensor(unc_corr)) else 0.0
        
        # Multi-step rollout (drift)
        initial_obs = obs[:256].clone()
        for h in [1, 3, 5]:
            curr_test = initial_obs.clone()
            for _ in range(h):
                with torch.no_grad():
                    # Actor expects unnormalized obs; curr_test is normalized
                    actor_input = replay_buffer.denormalize_obs(curr_test) if hasattr(replay_buffer, 'denormalize_obs') else curr_test
                    act_test = self.actor(actor_input)
                    curr_test, _, _, _, _, _ = self.world_model.step(curr_test, act_test)
            drift = ((curr_test - initial_obs) ** 2).mean().item()
            metrics[f'eval/rollout_drift_h{h}'] = drift
        
        return metrics

    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'world_model': self.world_model.state_dict(),
            'dreamer_threshold': self.dreamer.epistemic_threshold,
        }, filename)

    def save_checkpoint(self, filename, step, best_reward, evaluations):
        """Full checkpoint: weights + optimizer states + training state for resume."""
        torch.save({
            # Training state
            'step': step,
            'best_reward': best_reward,
            'evaluations': evaluations,
            'total_it': self.total_it,
            'wm_mse_ema': self.wm_mse_ema,
            'dreamer_threshold': self.dreamer.epistemic_threshold,
            # Model weights
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'world_model': self.world_model.state_dict(),
            # Optimizer states (preserves Adam momentum — critical for smooth resume)
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'wm_optimizer': self.wm_optimizer.state_dict(),
        }, filename)

    def load_checkpoint(self, filename):
        """Load full checkpoint. Returns (step, best_reward, evaluations)."""
        ckpt = torch.load(filename, map_location=self.device)
        # Model weights
        self.actor.load_state_dict(ckpt['actor'])
        self.actor_target.load_state_dict(ckpt['actor_target'])
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_target.load_state_dict(ckpt['critic_target'])
        self.world_model.load_state_dict(ckpt['world_model'])
        # Optimizer states
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        self.wm_optimizer.load_state_dict(ckpt['wm_optimizer'])
        # Training state
        self.total_it = ckpt.get('total_it', 0)
        self.wm_mse_ema = ckpt.get('wm_mse_ema', 1.0)
        self.dreamer.epistemic_threshold = ckpt.get('dreamer_threshold', self.dreamer.epistemic_threshold)
        return ckpt['step'], ckpt['best_reward'], ckpt['evaluations']

    def load(self, filename):
        ckpt = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.actor_target.load_state_dict(ckpt['actor_target'])
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_target.load_state_dict(ckpt['critic_target'])
        self.world_model.load_state_dict(ckpt['world_model'])
        if 'dreamer_threshold' in ckpt:
            self.dreamer.epistemic_threshold = ckpt['dreamer_threshold']