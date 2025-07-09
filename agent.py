# File: agent.py

import torch
import random
import numpy as np
from game import SnakeGame, Direction, Point
from collections import deque
from model import LinearQNetwork, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # controls exploration/randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # pop oldest memory when full
        self.model = LinearQNetwork(11, 256, 3)  # input size, hidden layer size, output size
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) 

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # pop left when max memory is reached
    
    def get_action(self, state, play_mode=False):
        """
        Get action from the model
        Args:
            state: current game state
            play_mode: if True, disable exploration (no random moves)
        """
        final_move = [0, 0, 0]  # [straight, right, left]
        
        if not play_mode:
            # Training mode: use epsilon-greedy exploration
            self.epsilon = 80 - self.n_games  # more games, less exploration
            
            if random.randint(0, 200) < self.epsilon:
                move = random.randint(0, 2)
                final_move[move] = 1
            else:
                state0 = torch.tensor(state, dtype=torch.float)
                prediction = self.model(state0)
                move = torch.argmax(prediction).item()
                # Add bounds checking for safety
                if 0 <= move <= 2:
                    final_move[move] = 1
                else:
                    # Fallback to random move if something goes wrong
                    move = random.randint(0, 2)
                    final_move[move] = 1
        else:
            # Play mode: always use the best predicted action (no exploration)
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            # Add bounds checking for safety
            if 0 <= move <= 2:
                final_move[move] = 1
            else:
                # Fallback to going straight if something goes wrong
                final_move[0] = 1
        
        return final_move  # return the action to take
    
    def train_long_memory(self):
        # FIXED: This was the main bug - wrong condition
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = list(self.memory)  # Convert deque to list for consistency

        if len(mini_sample) > 0:  # Only train if we have data
            states, actions, rewards, next_states, dones = zip(*mini_sample)
            self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_state(self, game):
        head = game.snake[0]  # Using snake[0] as in your original code
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location relative to snake head (IMPROVED)
            game.food.x < head.x,  # food left of head
            game.food.x > head.x,  # food right of head
            game.food.y < head.y,  # food above head
            game.food.y > head.y   # food below head
        ]

        return np.array(state, dtype=int)

    def load_model(self, file_name='model.pth'):
        """Load a saved model checkpoint"""
        return self.model.load(file_name)


def train():
    """Training function"""
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    agent = Agent()
    game = SnakeGame()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old, play_mode=False)  # Training mode
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done) #store in memory

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save('model.pth')

            print('Game', agent.n_games, 'Score:', score, 'Record:', record)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games

            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


def play_with_model(model_file='model.pth', num_games=10):
    """
    Play the game using a saved model
    Args:
        model_file: path to the saved model file
        num_games: number of games to play
    """
    agent = Agent()
    game = SnakeGame()
    
    # Load the saved model
    if not agent.load_model(model_file):
        print("Failed to load model. Make sure you have trained and saved a model first.")
        return
    
    print(f"Playing {num_games} games with loaded model...")
    scores = []
    
    for game_num in range(num_games):
        game.reset()
        
        while True:
            state = agent.get_state(game)
            action = agent.get_action(state, play_mode=True)  # Play mode (no exploration)
            reward, done, score = game.play_step(action)
            
            if done:
                print(f'Game {game_num + 1}: Score = {score}')
                scores.append(score)
                break
    
    print(f"\nResults:")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Best Score: {max(scores)}")
    print(f"Worst Score: {min(scores)}")


def play_human_vs_ai(model_file='model.pth'):
    """
    Play a single game where you can watch the AI play
    """
    agent = Agent()
    game = SnakeGame()
    
    # Load the saved model
    if not agent.load_model(model_file):
        print("Failed to load model. Make sure you have trained and saved a model first.")
        return
    
    print("Watching AI play... Press CTRL+C to stop")
    game.reset()
    
    try:
        while True:
            state = agent.get_state(game)
            action = agent.get_action(state, play_mode=True)
            reward, done, score = game.play_step(action)
            
            if done:
                print(f'Game Over! Final Score: {score}')
                game.reset()
    except KeyboardInterrupt:
        print("\nGame stopped by user")
        pygame.quit()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            train()
        elif sys.argv[1] == "play":
            model_file = sys.argv[2] if len(sys.argv) > 2 else 'model.pth'
            num_games = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            play_with_model(model_file, num_games)
        elif sys.argv[1] == "watch":
            model_file = sys.argv[2] if len(sys.argv) > 2 else 'model.pth'
            play_human_vs_ai(model_file)
        else:
            print("Usage:")
            print("  python agent.py train                    # Train the model")
            print("  python agent.py play [model_file] [n]    # Play n games with saved model")
            print("  python agent.py watch [model_file]       # Watch AI play continuously")
    else:
        # Default behavior: train
        train()