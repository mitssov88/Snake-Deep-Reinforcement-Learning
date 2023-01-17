import torch
import random
import numpy as np
from snake_game import BLOCK_SIZE, SnakeGame, Point, Direction
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100000 # 100,000 items in this memory available
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:
    def __init__(self):
        self.gameCount = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate (smaller than 1, usually around 0.8 or 0.9)
        self.memory = deque(maxlen=MAX_MEMORY) # pops the left when exceeded
        self.model = Linear_QNet(11, 256, 3) # 11 states, 256 hidden and 3 output
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)

    def get_state(self, game):        
        head = game.snake[0]

        headUp = Point(head.x, head.y - BLOCK_SIZE)
        headRight = Point(head.x + BLOCK_SIZE, head.y)
        headDown = Point(head.x, head.y + BLOCK_SIZE)
        headLeft = Point(head.x - BLOCK_SIZE, head.y)

        state = [
            # Danger in going Straight
            (game.direction == Direction.LEFT and game.is_collision(headLeft)) or 
            (game.direction == Direction.UP and game.is_collision(headUp)) or 
            (game.direction == Direction.RIGHT and game.is_collision(headRight)) or
            (game.direction == Direction.DOWN and game.is_collision(headDown)),

            # Danger in going Right
            (game.direction == Direction.LEFT and game.is_collision(headUp)) or 
            (game.direction == Direction.UP and game.is_collision(headRight)) or 
            (game.direction == Direction.RIGHT and game.is_collision(headDown)) or
            (game.direction == Direction.DOWN and game.is_collision(headLeft)),
            
            # Danger in going Left
            (game.direction == Direction.LEFT and game.is_collision(headDown)) or 
            (game.direction == Direction.UP and game.is_collision(headLeft)) or 
            (game.direction == Direction.RIGHT and game.is_collision(headUp)) or
            (game.direction == Direction.DOWN and game.is_collision(headRight)),

            # direction left
            1 if game.direction == Direction.LEFT else 0,
            
            # direction right
            1 if game.direction == Direction.RIGHT else 0,

            # direction up
            1 if game.direction == Direction.UP else 0,

            # direction down
            1 if game.direction == Direction.DOWN else 0,

            # # food left
            # 1 if game.food.x < game.head.x else 0,

            # # food right
            # 1 if game.food.x > game.head.x else 0,

            # # food up
            # 1 if game.food.y < game.head.y else 0,
            
            # # food down
            # 1 if game.food.y > game.head.y else 0
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

        
    def remember(self, state, action, reward, nextState, gameOver):
        # pop left if Max_memory is exceeded
        self.memory.append((state, action, reward, nextState, gameOver))

    def train_long_memory(self):
        # optimization after game ends
        # we grab 1000 samples aka 1 batch from memory
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # returns a list of tuples
        else:
            mini_sample = self.memory

        # put every state together, every action together, etc..
        states, actions, rewards, nextStates, gameOvers = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, nextStates, gameOvers)

        
    def train_short_memory(self, state, action, reward, nextState, gameOver):
        self.trainer.train_step(state, action, reward, nextState, gameOver)


    def get_action(self, state):
        # Tradeoff b/w exploration and explotation in Deep Learning. 
        # Sometimes we also want to make random moves to EXPLORE the environment 
        # But the better we get, the more we want to EXPLOIT our model
        # dependent on the number of games
        self.epsilon = 80 - self.gameCount
        finalMove = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon : 
            finalMove[random.randint(0, 2)] = 1
        # Initially: 40% chance of random move
        # As more games are played, chance of a random move DECREASES
        else:
            # Move based on our MODEL
            state0 = torch.tensor(state, dtype=torch.float) # turn the state into a tensor
            prediction = self.model(state0) 
            # take the max value of this prediction:
            finalMove[torch.argmax(prediction).item()] # .item is called to convert it into an int

        return finalMove


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    while True:
        # get old state
        state_old = agent.get_state(game)
        action = agent.get_action(state_old) # calls the model
        reward, game_over, score = game.play_step(action)
        state_new = agent.get_state(game)
        
        # train short memory
        agent.train_short_memory(state_old, action, reward, state_new, game_over)

        # remember, store this in our deque
        agent.remember(state_old, action, reward, state_new, game_over)

        if game_over:
            # train long memory a.k.a. Replay memory, experience memory i.e. trains on previous games
            game.reset()
            agent.gameCount += 1
            agent.train_long_memory()

            if record < score:
                record = score
                agent.model.save()
            print('Game #', agent.gameCount, 'Score:', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.gameCount
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    train()