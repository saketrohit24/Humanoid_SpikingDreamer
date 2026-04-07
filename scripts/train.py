#!/usr/bin/env python3
"""
Training script for Spiking Dreamer.

Usage:
    python train.py --config configs/halfcheetah.yaml
    python train.py --env HalfCheetah-v4 --seed 42
    python train.py --no-dreaming  # Baseline mode
    python train.py --resume       # Auto-resume from latest checkpoint
"""

import os
import sys
import glob
import random
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spiking_dreamer import TD3_SpikingDreamer, ReplayBuffer, eval_policy, make_env


def find_latest_checkpoint(suffix, env_name, seed):
    """Return the path of the most recent periodic checkpoint, or None."""
    pattern = f"checkpoints/{suffix}_{env_name}_{seed}_step*.pt"
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def cleanup_old_checkpoints(suffix, env_name, seed, keep=3):
    """Delete old periodic checkpoints, keeping only the most recent `keep`."""
    pattern = f"checkpoints/{suffix}_{env_name}_{seed}_step*.pt"
    files = sorted(glob.glob(pattern))
    for f in files[:-keep]:
        os.remove(f)
        print(f"  Removed old checkpoint: {f}")


def load_config(config_path: str, default_path: str = "configs/default.yaml") -> dict:
    """Load config from YAML file, with defaults."""
    # Load defaults
    with open(default_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with specific config
    if config_path and config_path != default_path:
        with open(config_path, 'r') as f:
            overrides = yaml.safe_load(f)
        config.update(overrides)
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Train Spiking Dreamer")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--env", type=str, default=None,
                        help="Override environment name")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--no-dreaming", action="store_true",
                        help="Disable dreaming (baseline mode)")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="spiking-dreamer",
                        help="W&B project name")
    parser.add_argument("--resume", action="store_true",
                        help="Auto-resume from latest periodic checkpoint")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply CLI overrides
    if args.env:
        config["env_name"] = args.env
    if args.seed is not None:
        config["seed"] = args.seed
    if args.no_dreaming:
        config["enable_dreaming"] = False
    
    print("=" * 70)
    print("Spiking Dreamer Training")
    print("=" * 70)
    
    env_name = config["env_name"]
    seed = config["seed"]
    
    # Environment setup
    env = make_env(env_name)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Add dimensions to config
    config["obs_dim"] = state_dim
    config["act_dim"] = action_dim
    
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    
    print(f"Environment: {env_name}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Device: {device}")
    print(f"Dreaming: {'ENABLED' if config['enable_dreaming'] else 'DISABLED'}")
    if config['enable_dreaming']:
        print(f"  Ensemble size: {config['num_ensemble']}")
        print(f"  Dream start: {config['dream_start_step']}")
        print(f"  Dream horizon: {config['dream_horizon']}")
    
    # Initialize
    replay_buffer = ReplayBuffer(state_dim, action_dim, config["replay_size"], device=device)
    policy = TD3_SpikingDreamer(state_dim, action_dim, max_action, device, config, replay_buffer)

    # Logging
    suffix = "spiking_dream" if config["enable_dreaming"] else "baseline"
    log_dir = f"runs/popsan_{suffix}_{env_name}_{seed}"
    writer = SummaryWriter(log_dir)
    os.makedirs("checkpoints", exist_ok=True)

    # --- Resume from checkpoint if requested ---
    start_t = 0
    best_reward = float('-inf')
    evaluations = []

    if args.resume:
        ckpt_path = find_latest_checkpoint(suffix, env_name, seed)
        if ckpt_path:
            start_t, best_reward, evaluations = policy.load_checkpoint(ckpt_path)
            print(f"Resumed from {ckpt_path} at step {start_t}")
            print(f"  Best reward so far: {best_reward:.1f}")
        else:
            print("No checkpoint found — starting fresh.")

    if not evaluations:
        evaluations = [eval_policy(policy, env_name, seed)]
        print(f"Initial eval: {evaluations[-1][0]:.1f}")
    print("-" * 70)

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{suffix}_{env_name}_{seed}",
            config=config,
            resume="allow",
        )

    state, _ = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    # Accumulators
    critic_loss_acc = 0.0
    actor_loss_acc = 0.0
    train_count = 0
    last_dream_metrics = {}

    checkpoint_freq = config.get("checkpoint_freq", 100000)
    total_t = int(config["total_steps"] + config["start_timesteps"])

    for t in range(start_t, total_t):
        episode_timesteps += 1

        # Warmup: random actions until buffer has enough data (handles fresh + resumed starts)
        if replay_buffer.size < config["start_timesteps"]:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(state)
                + np.random.normal(0, max_action * config["expl_noise"], size=action_dim)
            ).clip(-max_action, max_action)

        next_state, reward, d1, d2, _ = env.step(action)
        done = d1 or d2
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # NaN guard: MuJoCo can return NaN contact forces on degenerate states
        if np.isnan(next_state).any() or np.isnan(reward):
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            continue

        replay_buffer.add(state, action, next_state, reward, done_bool, is_dream=False)
        state = next_state
        episode_reward += reward

        # Training — only once buffer is large enough
        if replay_buffer.size >= config["batch_size"] and replay_buffer.size >= config["start_timesteps"]:
            if config["enable_dreaming"]:
                for _ in range(config.get("wm_updates_per_step", 1)):
                    policy.train_world_model(replay_buffer, config["batch_size"])

            c_loss, a_loss = policy.train(replay_buffer, config["batch_size"])
            critic_loss_acc += c_loss
            actor_loss_acc += a_loss
            train_count += 1

            # Dreaming phase
            if config["enable_dreaming"] and t >= config["dream_start_step"] and t % config["dream_freq"] == 0:
                dreams_added, phase_metrics_list = policy.dream_phase(replay_buffer)
                if phase_metrics_list:
                    last_dream_metrics = phase_metrics_list[-1]
                    if dreams_added == 0:
                        mean_unc = last_dream_metrics.get('mean_uncertainty', -1)
                        thresh = last_dream_metrics.get('epistemic_threshold', -1)
                        print(f"  [Dream] 0 accepted — mean_unc={mean_unc:.4f} threshold={thresh:.4f}")
        
        if done:
            dream_buf_size = policy.dream_buffer.size if config["enable_dreaming"] else 0
            print(f"T:{t+1} Ep:{episode_num+1} R:{episode_reward:.0f} "
                  f"RealBuf:{replay_buffer.size} DreamBuf:{dream_buf_size}")
            writer.add_scalar("train/episode_reward", episode_reward, t)
            
            if args.wandb:
                wandb.log({
                    "train/episode_reward": episode_reward,
                    "buffer/real_size": replay_buffer.size,
                    "buffer/dream_size": dream_buf_size,
                }, step=t)
            
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        
        # Evaluation
        if (t + 1) % config["eval_freq"] == 0:
            eval_reward, eval_std = eval_policy(policy, env_name, seed)
            evaluations.append((eval_reward, eval_std))
            writer.add_scalar("eval/reward", eval_reward, t)
            
            # Log metrics
            if train_count > 0:
                writer.add_scalar("train/critic_loss", critic_loss_acc / train_count, t)
                writer.add_scalar("train/actor_loss", actor_loss_acc / train_count, t)
            
            if config["enable_dreaming"]:
                wm_metrics = policy.get_wm_metrics()
                for k, v in wm_metrics.items():
                    writer.add_scalar(k, v, t)
                
                eval_wm_metrics = policy.evaluate_world_model(replay_buffer)
                for k, v in eval_wm_metrics.items():
                    writer.add_scalar(k, v, t)
                
                if args.wandb:
                    wandb_logs = {
                        "eval/reward": eval_reward,
                        "train/critic_loss": critic_loss_acc / max(1, train_count),
                        **wm_metrics,
                        **eval_wm_metrics,
                    }
                    wandb.log(wandb_logs, step=t)
                
                print(f"Eval @ {t + 1}: {eval_reward:.1f} ± {eval_std:.1f} | Best: {best_reward:.1f}")
                print(f"  WM: MSE={wm_metrics.get('wm/state_mse', 0):.4f} | "
                      f"R²={eval_wm_metrics.get('eval/r2_score', 0):.4f}")
            else:
                print(f"Eval @ {t + 1}: {eval_reward:.1f} ± {eval_std:.1f}")
                if args.wandb:
                    wandb.log({"eval/reward": eval_reward}, step=t)
            
            print("-" * 70)
            
            if eval_reward > best_reward:
                best_reward = eval_reward
                policy.save(f"checkpoints/{suffix}_{env_name}_{seed}_best.pt")
                print(f"  *** New best! Saved best checkpoint ***")

            critic_loss_acc = 0.0
            actor_loss_acc = 0.0
            train_count = 0

        # Periodic checkpoint — save every checkpoint_freq steps so we can always resume
        if (t + 1) % checkpoint_freq == 0:
            ckpt_path = f"checkpoints/{suffix}_{env_name}_{seed}_step{t+1:08d}.pt"
            policy.save_checkpoint(ckpt_path, t + 1, best_reward, evaluations)
            print(f"[Checkpoint] Saved {ckpt_path}")
            cleanup_old_checkpoints(suffix, env_name, seed, keep=3)

    # Final save
    policy.save_checkpoint(
        f"checkpoints/{suffix}_{env_name}_{seed}_final.pt",
        total_t, best_reward, evaluations
    )
    
    print("\n" + "=" * 70)
    print(f"Training Complete! Best reward: {best_reward:.1f}")
    if config["enable_dreaming"]:
        print(f"Dreams generated: {policy.dreamer.stats['total_generated']}")
        print(f"Dreams accepted: {policy.dreamer.stats['total_accepted']}")
    print("=" * 70)
    
    if args.wandb:
        wandb.finish()
    
    env.close()
    writer.close()


if __name__ == "__main__":
    # Set precision for tensor cores
    torch.set_float32_matmul_precision('high')
    main()
