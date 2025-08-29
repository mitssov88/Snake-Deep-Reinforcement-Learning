import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

WHITE = (255, 255, 255)
RED = (252, 3, 173)
BLUE1 = (3, 252, 248)
BLACK = (0,0,0)
BLOCK_SIZE = 20
SPEED = 40

pygame.init()
font = pygame.font.Font(None, 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
class SnakeGame:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.iteration = 0
        self.reset()
        
            
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self.iteration = 0
        self._place_food()

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        # food cannot be inside the snake
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):
        # collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action) # update the head

        self.snake.insert(0, self.head)
        self.iteration += 1

        # check if game over
        game_over = False
        reward = 0
        if self.is_collision() or self.iteration > 500*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        # place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        # hits boundary
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        for i, pt in enumerate(self.snake):
            if i == 0:  # Head - filled in (no inner rectangle)
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            else:  # Body - normal color with hollow center
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLACK, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        x = self.head.x
        y = self.head.y

        clockwise = [Direction.LEFT, Direction.UP, Direction.RIGHT, Direction.DOWN]
        idx = clockwise.index(self.direction)
        if np.array_equal(action, [0, 1, 0]): # right
            idx += 1
        elif np.array_equal(action, [0, 0, 1]): # left
            idx -= 1
        
        self.direction = clockwise[idx % 4]

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)

# Point is a tuple of x and y coordinates
Point = namedtuple('Point', 'x, y')