import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGame, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000 # stores a set of data regarding every decision made using the model
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate, decreases value of reward over time
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3) # kinda a subset of QTrainer since QTrainer calls model to do the forward pass
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
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
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)
    
    def get_action(self, state):
        self.epsilon = 100 - self.n_games
        if self.epsilon < 1: 
            self.epsilon = 1
        
        final_move = [0,0,0]

        if random.randint(0, 200) < self.epsilon: # make a random move. The likelihood of this occurring decreases
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def train_short_memory(self, state, action, reward, next_state, gameOver):
        self.trainer.train_step(state, action, reward, next_state, gameOver)

    def remember(self, state, action, reward, next_state, gameOver):
        self.memory.append((state, action, reward, next_state, gameOver)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, gameOvers = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, gameOvers)

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    while True:
        state_old = agent.get_state(game) # get old state
        action = agent.get_action(state_old) # calls the model
        reward, game_over, score = game.play_step(action) # update the snake's position and retrieve any new events
        state_new = agent.get_state(game) # get new state
        
        # train short memory
        agent.train_short_memory(state_old, action, reward, state_new, game_over)

        # remember, store this in our deque
        agent.remember(state_old, action, reward, state_new, game_over)

        if game_over:
            game.reset()
            agent.n_games += 1

            # train long memory a.k.a. Replay memory, experience memory i.e. trains on previous decisions
            agent.train_long_memory()

            if record < score:
                record = score
                agent.model.save()

            print('Game #', agent.n_games, 'Score:', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    train()