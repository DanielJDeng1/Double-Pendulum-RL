# Double Pendulum RL

A small reinforcement learning project using PyTorch PPO to train an agent on MuJoCo's `InvertedDoublePendulum-v5`.

The main goal was to get PPO working from scratch and see how well it could learn to keep the double pendulum balanced. The project also includes a simple Tkinter dashboard for starting training runs and looking at saved checkpoints without having to manage everything from the command line.

## Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/DanielJDeng1/Double-Pendulum-RL.git
cd Double-Pendulum-RL

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Running the Project

### GUI

If you want to use the dashboard:

```bash
python gui_launcher.py
```

The GUI can be used to start training runs and work with saved models.

### Training from the command line

example:

```bash
python train.py --run-name ppo_double_pendulum --total-steps 1000000 --num-envs 8
```

this starts a PPO run using 8 environments in parallel.

### Watching a checkpoint

once a model has been trained, load it into the viewer:

```bash
python live_view.py --checkpoint models/ppo_double_pendulum.pth
```

### Evaluating models

To run several evaluation episodes:

```bash
python evaluate.py --checkpoint models/ppo_double_pendulum.pth --episodes 20 --render
```

## Project Structure

```text
Double-Pendulum-RL/
├── agent.py          # agent network
├── env_utils.py      # environment setup and wrappers
├── train.py          # PPO training loop
├── evaluate.py       # batch evaluation
├── live_view.py      # checkpoint viewer
├── gui_launcher.py   # Tkinter training dashboard
└── view_sim.py       # MuJoCo simulation viewer
```
