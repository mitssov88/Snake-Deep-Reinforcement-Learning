# Snake Deep Q-Learning Reinforcement Learning
## Overview

This project implements a Deep Q-Network (DQN) agent that learns to play the classic Snake game using Deep Reinforcement Learning. The agent is trained to navigate the game environment, eat apples to grow, and avoid collisions with walls or itself. After training, the model demonstrates superhuman performance, consistently achieving high scores and outperforming human players.

Built with Python, PyTorch, PyGame, and Matplotlib.

## Features

- **Deep Q-Network (DQN)**: A reinforcement learning algorithm that uses a neural network to approximate the Q-value function.
- **Self-Learning Agent**: The model learns optimal strategies through trial and error without any predefined rules.
- **Game Visualization**: Real-time rendering of the Snake game during training and playback.
- **Performance Metrics**: Plots of scores, episode lengths, and learning curves.
- **Demo Videos**: Pre-recorded gameplay videos (.mp4 files) showcasing the trained agent's performance.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/mitssov88/Snake-Deep-Reinforcement-Learning.git
   cd Snake-Deep-Reinforcement-Learning
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Agent
To train the model and watch the agent play:
```
python main.py
```

- The script will start training the DQN agent.
- During training, the game window will display the agent's actions in real-time.
- Training progress (e.g., episode scores) will be logged to the console and visualized with Matplotlib plots.

### Viewing Demos
Download the `.mp4` files from the repository to see the trained agent in action without running the code. These videos demonstrate the agent's ability to achieve high scores efficiently.
