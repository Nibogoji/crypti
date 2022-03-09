"""
Il succo é approssimare la policy con un deep neural net.
La natura stocastica del risultato funge anche da explore/exploit
Puo essere usato in spazi continui perche apprende dal esperienza

La funzione che approssima la policy come un campo vettoriale 4d, che
per ogni punto a,s,theta( parametro NN ) da un'indicazione sul cosa fare

Il cosa fare deve essere la direzione che massimizza la funzione 3d Q(s,a)
Ogni policy ha le sua Q(s,a) per ogni a dato s.

Riordinre a per similitudine rispetto al massimizzare q come farebbe xgboost?
se 2 a con simile q sono distanti nello spazio la policy fa piu fatica a decidere?

Algo : Rinforce
"""
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(),lr = lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class PolicyGradientAgent():
    def __init__(self, lr, input_dims, gamma = 0.99, n_actions = 4):
        self.gamma = gamma
        self.lr = lr
        self.reward_memory = []
        self.action_memory = []

        self.policy = PolicyNetwork(self.lr,input_dims,n_actions)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.policy.device) # il tensor deve essere nella stessa scheda di processo
        probabilities = F.softmax(self.policy.forward(state))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def learn(self):

        self.policy.optimizer.zero_grad()
        # G_t = R_t+1 +gamma * R_t+2 + gamma**2 * R_t+3

        G = np.zeros_like(self.reward_memory)

        for t in range(len(self.reward_memory)):

            G_sum = 0
            discount = 1

            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k]*discount
                discount *= self.gamma
            G[t] = G_sum
        G = T.tensor(G,dtype=T.float).to(self.policy.device)

        loss = 0

        for g, logprob in zip(G, self.action_memory):
            loss += -g*logprob
        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []







