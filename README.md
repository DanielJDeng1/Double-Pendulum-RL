# Double Pendulum RL — PPO on InvertedDoublePendulum-v5

A PyTorch PPO agent for the MuJoCo `InvertedDoublePendulum-v5` Gymnasium
environment, set up for Windows 11 + an NVIDIA RTX 5060 Laptop GPU
(Blackwell, `sm_120`).

## 1. Setup (Windows PowerShell)

```powershell
cd double_pendulum_rl
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

# CUDA 12.8 nightly wheel FIRST — see requirements.txt for why this matters
pip install --pre torch torchvision torchaudio `
    --index-url https://download.pytorch.org/whl/nightly/cu128

pip install -r requirements.txt

# sanity check
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `torch.cuda.is_available()` is `True` but training still falls back to
CPU with a "no kernel image is available" error, your wheel predates
`sm_120` kernel support — reinstall from the nightly index above.

## 2. Render smoke-test

Before training, confirm MuJoCo can open a window on your system:

```powershell
python view_sim.py
```

This drives the pendulum with random actions so you can confirm the
viewer renders. No trained model needed.

## 3. Train

```powershell
python train.py --total-steps 1000000 --num-envs 8
```

Runs headless (no rendering) for speed. Checkpoints save to
`models/ppo_double_pendulum.pth` every 10 updates. Watch progress with:

```powershell
tensorboard --logdir runs
```

Key flags: `--total-steps`, `--num-envs`, `--num-steps`, `--learning-rate`,
`--seed`, `--run-name`. See `train.py`'s `PPOConfig` dataclass for every
hyperparameter (GAE lambda, clip epsilon, entropy/value coefficients, etc.).

## 4. Watch the trained policy

```powershell
python view_sim.py --checkpoint models\ppo_double_pendulum.pth
```

## 5. Evaluate

```powershell
python evaluate.py --checkpoint models\ppo_double_pendulum.pth --episodes 50 --csv-out results\eval.csv
```

Add `--render` to watch while it evaluates, or `--stochastic` to sample
from the policy distribution instead of using its mean action.

## Files

| File | Purpose |
|---|---|
| `agent.py` | `ActorCritic` network: Gaussian policy head + separate value head |
| `train.py` | Headless PPO training loop (GAE, clipped surrogate loss, TensorBoard) |
| `view_sim.py` | Interactive MuJoCo viewer — random policy or a loaded checkpoint |
| `evaluate.py` | Batch evaluation with return/length statistics and optional CSV export |
| `requirements.txt` | Dependencies, with the Blackwell/CUDA 12.8 nightly install note |

## Algorithm notes

- **Policy**: diagonal Gaussian, state-dependent mean, state-independent
  learned log-std (standard PPO continuous-control setup).
- **Advantage estimation**: Generalized Advantage Estimation (GAE-λ).
- **Objective**: PPO clipped surrogate loss + value loss + entropy bonus.
- **Vectorization**: `gymnasium.vector.SyncVectorEnv` across `--num-envs`
  parallel environment copies for faster rollout collection on CPU while
  the network forward/backward passes run on GPU.

This is a from-scratch reference implementation (not `stable-baselines3`)
so every piece — rollout buffer, GAE, PPO-clip, checkpointing — is visible
and editable in `train.py`.
