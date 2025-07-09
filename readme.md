# 🐍 Snake AI - Deep Q-Learning Agent

A sophisticated Snake game AI that learns to play using Deep Q-Learning (DQN). Watch as the AI progressively gets better at the classic Snake game through reinforcement learning!

## 🌟 Features

- **Deep Q-Learning Implementation**: Uses a neural network to learn optimal moves
- **Real-time Training Visualization**: Watch the AI improve with live score plotting
- **Model Persistence**: Save and load trained models
- **Multiple Play Modes**: Train, test, or watch the AI play
- **Performance Statistics**: Detailed analysis of AI performance
- **Customizable Parameters**: Easy to modify network architecture and training parameters

## 🎮 Demo

The AI starts by making random moves but quickly learns to:
- Avoid walls and its own body
- Navigate efficiently toward food
- Maximize score by growing as long as possible
- Achieve consistent high scores after training

## 📋 Requirements

```
pygame>=2.0.0
torch>=1.9.0
numpy>=1.19.0
matplotlib>=3.3.0
```

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SabadModi/RL_Snake_Game
   cd RL_Snake_Game
   ```

2. **Install dependencies**:
   ```bash
   pip install pygame torch numpy matplotlib
   ```

3. **Run the game**:
   ```bash
   python agent.py train
   ```

## 📁 Project Structure

```
snake-ai/
├── agent.py          # Main AI agent with training and playing logic
├── game.py           # Snake game implementation
├── model.py          # Neural network architecture and training
├── helper.py         # Plotting utilities for training visualization
├── play.py           # Simple script to play with trained models
├── model/            # Directory for saved model checkpoints
│   └── model.pth     # Trained model weights
└── README.md         # This file
```

## 🎯 How It Works

### Neural Network Architecture
- **Input Layer**: 11 features representing game state
  - Danger detection (straight, right, left)
  - Current direction (4 boolean values)
  - Food location relative to head (4 boolean values)
- **Hidden Layers**: 2 layers with 256 neurons each
- **Output Layer**: 3 actions (straight, right, left)

### State Representation
The AI perceives the game through 11 boolean features:
```python
[danger_straight, danger_right, danger_left, 
 dir_left, dir_right, dir_up, dir_down,
 food_left, food_right, food_up, food_down]
```

### Reward System
- **+10**: Eating food
- **-10**: Game over (collision)
- **0**: Normal move

### Training Process
1. **Exploration vs Exploitation**: Uses epsilon-greedy strategy
2. **Experience Replay**: Stores experiences in memory buffer
3. **Target Network**: Stabilizes training with delayed updates
4. **Batch Learning**: Trains on random batches from memory

## 🎮 Usage

### Training the AI

Start training from scratch:
```bash
python agent.py train
```

The AI will:
- Display the game window with real-time gameplay
- Show a live plot of scores and mean scores
- Automatically save the best model to `model/model.pth`
- Print progress: `Game 1 Score: 0 Record: 0`

### Playing with Trained Model

#### Option 1: Command Line Interface
```bash
# Play 10 games and see statistics
python agent.py play

# Play 5 games with specific model
python agent.py play model.pth 5

# Watch AI play continuously
python agent.py watch

# Watch with specific model
python agent.py watch model.pth
```

#### Option 2: Interactive Script
```bash
python play.py
```

This provides a user-friendly menu with options to:
- Watch the AI play continuously
- Run multiple games with detailed statistics

### Example Output
```
🐍 Snake AI Player 🐍
==============================
Loading model from model.pth...
✅ Model loaded successfully!

🏃 Running 10 games...
Game  1/10: Score =  8
Game  2/10: Score = 12
Game  3/10: Score = 15
...

📈 Statistics:
   Average Score: 11.40
   Best Score:    18
   Worst Score:   3
   Total Games:   10

📊 Score Distribution:
   0-5:      1 games (10.0%)
   6-10:     4 games (40.0%)
   11-20:    5 games (50.0%)
   21+:      0 games (0.0%)
```

## ⚙️ Configuration

### Training Parameters
Modify these in `agent.py`:
```python
MAX_MEMORY = 100_000    # Experience replay buffer size
BATCH_SIZE = 1000       # Training batch size
LR = 0.001             # Learning rate
```

### Game Parameters
Modify these in `game.py`:
```python
BLOCK_SIZE = 20         # Size of each game block
SPEED = 20             # Game speed (FPS)
```

### Network Architecture
Modify in `model.py`:
```python
self.model = LinearQNetwork(11, 256, 3)  # input, hidden, output
```