import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) # setting up the model. input size contains the number of features. 11 in our case
        self.linear2 = nn.Linear(hidden_size, output_size) # outputs 3 values, max is the direction we move in

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name) # save the state dict to the directory

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # updates the weights after backward pass that calculated dL/dweights_i
        self.criterion = nn.MSELoss() # this is the Loss calculation

    def train_step(self, state, action, reward, newState, gameOver):
        state = torch.tensor(state, dtype=torch.float)
        newState = torch.tensor(newState, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1: # we want to turn lists of [0, 0, 0, 1] to matrices with 1 row [[0, 0, 0, 1]] i.e. add dimension
            state = torch.unsqueeze(state, 0)
            newState = torch.unsqueeze(newState, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            gameOver = (gameOver, )
        
        # 1: predicted Q values with current state
        prediction = self.model(state) # Returns the actions
        target = prediction.clone()

        for idx in range(len(gameOver)):
            Q_new = reward[idx]
            if not gameOver[idx]: # 2: Q_new = reward + gamma * max(next_predicted Q value) -> only do this if not gameOver
                Q_new = reward[idx] + self.gamma * torch.max(self.model(newState[idx]))

            # predictions[argmax(action)] = Q_new
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad() # clear the weights gradients after the optimization step (I guess this works because if you loop, this goes after 143)
        loss = self.criterion(target, prediction) # Mean Squared Error of target and rorid
        loss.backward() # calculate gradients. access dL/dweight_i by calling weight_i.grad()
        self.optimizer.step() # updates parameters a.k.a. weights