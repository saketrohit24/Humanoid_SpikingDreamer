#!/usr/bin/env python3
"""
GPU-optimized Replay Buffer with dream tracking.
"""

import torch
import numpy as np


class ReplayBuffer:
    """
    GPU Replay Buffer with:
    - Pre-allocated tensors on GPU
    - Dream/real transition tracking
    - Observation normalization
    - Batch add/sample operations
    """
    
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cuda'):
        self.device = torch.device(device)
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state_dim = state_dim
        
        # Pre-allocate TENSORS on GPU
        self.state = torch.zeros((max_size, state_dim), device=self.device)
        self.action = torch.zeros((max_size, action_dim), device=self.device)
        self.next_state = torch.zeros((max_size, state_dim), device=self.device)
        self.reward = torch.zeros((max_size, 1), device=self.device)
        self.not_done = torch.zeros((max_size, 1), device=self.device)
        self.is_dream = torch.zeros(max_size, dtype=torch.bool, device=self.device)
        
        # Separate counters
        self.real_count = 0
        self.dream_count = 0
        
        # Normalization statistics (kept on GPU)
        self.mean = torch.zeros(state_dim, device=self.device)
        self.var = torch.ones(state_dim, device=self.device)
        self.clip_limit = 5.0
        self.norm_update_interval = 100

    def add(self, state, action, next_state, reward, done, is_dream=False):
        # Input validation and transfer
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device, dtype=torch.float32)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, device=self.device, dtype=torch.float32)
            
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = float(reward)
        self.not_done[self.ptr] = 1. - float(done)
        self.is_dream[self.ptr] = is_dream
        
        if is_dream:
            self.dream_count += 1
        else:
            self.real_count += 1
            if self.real_count % self.norm_update_interval == 0:
                self._update_normalization_stats()
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, states, actions, next_states, rewards, dones, is_dream=False):
        batch_size = len(states)
        
        # Ensure inputs are tensors on device
        if not isinstance(states, torch.Tensor): states = torch.tensor(states, device=self.device)
        if not isinstance(actions, torch.Tensor): actions = torch.tensor(actions, device=self.device)
        if not isinstance(next_states, torch.Tensor): next_states = torch.tensor(next_states, device=self.device)
        if not isinstance(rewards, torch.Tensor): rewards = torch.tensor(rewards, device=self.device)
        if not isinstance(dones, torch.Tensor): dones = torch.tensor(dones, device=self.device)
        
        if self.ptr + batch_size <= self.max_size:
            self.state[self.ptr:self.ptr+batch_size] = states
            self.action[self.ptr:self.ptr+batch_size] = actions
            self.next_state[self.ptr:self.ptr+batch_size] = next_states
            self.reward[self.ptr:self.ptr+batch_size] = rewards.view(-1, 1)
            self.not_done[self.ptr:self.ptr+batch_size] = 1. - dones.view(-1, 1)
            self.is_dream[self.ptr:self.ptr+batch_size] = is_dream
            
            self.ptr = (self.ptr + batch_size) % self.max_size
            self.size = min(self.size + batch_size, self.max_size)
        else:
            space_left = self.max_size - self.ptr
            self.add_batch(states[:space_left], actions[:space_left], next_states[:space_left], 
                          rewards[:space_left], dones[:space_left], is_dream)
            self.add_batch(states[space_left:], actions[space_left:], next_states[space_left:], 
                          rewards[space_left:], dones[space_left:], is_dream)
        
        if is_dream:
            self.dream_count += batch_size
        else:
            self.real_count += batch_size
    
    def _update_normalization_stats(self):
        if self.size == 0: return
        
        real_mask = ~self.is_dream[:self.size]
        if real_mask.sum() == 0: return
            
        data = self.state[:self.size][real_mask]
        self.mean = data.mean(dim=0)
        self.var = data.var(dim=0) + 1e-8
    
    def normalize_obs(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
        elif obs.device != self.device:
            obs = obs.to(self.device)
            
        normalized = (obs - self.mean) / torch.sqrt(self.var)
        return torch.clamp(normalized, -self.clip_limit, self.clip_limit)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.not_done[ind]
        )
    
    def sample_real_only(self, batch_size):
        real_mask = ~self.is_dream[:self.size]
        real_idx = torch.where(real_mask)[0]
        
        if len(real_idx) == 0:
            return self.sample(batch_size)
            
        idx_in_real = torch.randint(0, len(real_idx), (min(batch_size, len(real_idx)),), device=self.device)
        ind = real_idx[idx_in_real]
        
        state_norm = self.normalize_obs(self.state[ind])
        next_state_norm = self.normalize_obs(self.next_state[ind])
        
        return (
            state_norm,
            self.action[ind],
            next_state_norm,
            self.reward[ind],
            self.not_done[ind]
        )
    
    def sample_states_for_dreaming(self, batch_size, device=None):
        real_mask = ~self.is_dream[:self.size]
        real_idx = torch.where(real_mask)[0]

        if len(real_idx) == 0:
            ind = torch.randint(0, self.size, (batch_size,), device=self.device)
            return self.normalize_obs(self.state[ind])

        # 60% from most recent 20% of real transitions (on-policy starts),
        # 40% uniform over all real transitions (coverage).
        recent_k = max(1, len(real_idx) // 5)
        recent_idx = real_idx[-recent_k:]

        n_recent = int(batch_size * 0.6)
        n_uniform = batch_size - n_recent

        recent_sample = recent_idx[torch.randint(0, len(recent_idx), (n_recent,), device=self.device)]
        uniform_sample = real_idx[torch.randint(0, len(real_idx), (n_uniform,), device=self.device)]
        ind = torch.cat([recent_sample, uniform_sample])

        return self.normalize_obs(self.state[ind])
        
    def denormalize_obs(self, obs):
        if not isinstance(obs, torch.Tensor):
            return obs * np.sqrt(self.var.cpu().numpy()) + self.mean.cpu().numpy()
        return obs * torch.sqrt(self.var) + self.mean
        
    def store(self, state, action, reward, next_state, done):
        self.add(state, action, next_state, reward, done, is_dream=True)
    
    def get_dream_ratio(self):
        if self.size == 0: return 0.0
        return self.is_dream[:self.size].float().mean().item()
    
    def get_stats(self):
        return {
            'size': self.size,
            'real_count': self.real_count,
            'dream_count': self.dream_count,
            'dream_ratio': self.get_dream_ratio(),
        }
