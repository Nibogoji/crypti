import os
import tensorflow.keras as tfk
from tensorflow.keras.layers import Dense


class ActorCriticNet(tfk.Model):
    def __init__(self, n_actions, fc1_dim = 1024, fc2_dim = 512,
                 name = 'actor_critic', checkpoint_dir = 'RL_actor_crytic/actor_critic'):
        super(ActorCriticNet,self).__init__()

        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,name+'_ac')

        self.fc1 = Dense(self.fc1_dim,activation = 'relu')
        self.fc2 = Dense(self.fc2_dim,activation = 'relu')
        self.v = Dense(1,activation=None)
        self.pi = Dense(n_actions,activation='softmax')

    def call(self, state):

        value = self.fc1(state)
        value = self.fc2(value)

        v = self.v(value)
        pi = self.pi(value)

        return v, pi



