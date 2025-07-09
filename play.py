# File: play.py

import pygame
import numpy as np
from agent import Agent
from game import SnakeGame

def main():
    """
    Simple script to load and play with a trained model
    """
    print("🐍 Snake AI Player 🐍")
    print("=" * 30)
    
    # Initialize agent and game
    agent = Agent()
    game = SnakeGame()
    
    # Try to load the model
    model_file = 'model.pth'
    print(f"Loading model from {model_file}...")
    
    if not agent.load_model(model_file):
        print("❌ No saved model found!")
        print("Please train the model first by running: python agent.py train")
        return
    
    print("✅ Model loaded successfully!")
    print("\nOptions:")
    print("1. Watch AI play one continuous game")
    print("2. Run multiple games and see statistics")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        watch_ai_play(agent, game)
    elif choice == "2":
        run_multiple_games(agent, game)
    else:
        print("Invalid choice. Running continuous play...")
        watch_ai_play(agent, game)


def watch_ai_play(agent, game):
    """Watch the AI play continuously"""
    print("\n🎮 Watching AI play...")
    print("Press CTRL+C or close the window to stop")
    
    game_count = 0
    total_score = 0
    
    try:
        while True:
            game.reset()
            game_count += 1
            
            while True:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                
                state = agent.get_state(game)
                action = agent.get_action(state, play_mode=True)
                reward, done, score = game.play_step(action)
                
                if done:
                    total_score += score
                    avg_score = total_score / game_count
                    print(f"Game {game_count}: Score = {score} | Average = {avg_score:.2f}")
                    break
                    
    except KeyboardInterrupt:
        print(f"\n🏁 Stopped after {game_count} games")
        if game_count > 0:
            print(f"📊 Average score: {total_score / game_count:.2f}")
        pygame.quit()


def run_multiple_games(agent, game):
    """Run multiple games and show statistics"""
    try:
        num_games = int(input("How many games to play? (default: 10): ") or "10")
    except ValueError:
        num_games = 10
    
    print(f"\n🏃 Running {num_games} games...")
    scores = []
    
    for i in range(num_games):
        game.reset()
        
        while True:
            state = agent.get_state(game)
            action = agent.get_action(state, play_mode=True)
            reward, done, score = game.play_step(action)
            
            if done:
                scores.append(score)
                print(f"Game {i+1:2d}/{num_games}: Score = {score:2d}")
                break
    
    # Show statistics
    print(f"\n📈 Statistics:")
    print(f"   Average Score: {np.mean(scores):.2f}")
    print(f"   Best Score:    {max(scores)}")
    print(f"   Worst Score:   {min(scores)}")
    print(f"   Total Games:   {len(scores)}")
    
    # Score distribution
    score_ranges = [(0, 5), (6, 10), (11, 20), (21, float('inf'))]
    print(f"\n📊 Score Distribution:")
    for min_score, max_score in score_ranges:
        if max_score == float('inf'):
            count = sum(1 for s in scores if s >= min_score)
            print(f"   {min_score}+:     {count:2d} games ({count/len(scores)*100:.1f}%)")
        else:
            count = sum(1 for s in scores if min_score <= s <= max_score)
            print(f"   {min_score}-{max_score}:     {count:2d} games ({count/len(scores)*100:.1f}%)")


if __name__ == "__main__":
    main()