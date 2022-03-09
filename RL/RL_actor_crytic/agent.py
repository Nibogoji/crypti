import tensorflow as tf
from tensorflow import optimizers
import tensorflow_probability as tfp
from RL.RL_actor_crytic import ActorCriticNet

class Agent:
    def __init__(self, alpha = 0.0003, gamma = 0.99, n_actions = 2):

        self.gamma = gamma
        self.n_actions = n_actions
        self.alpha = alpha
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]

        self.actor_critic = ActorCriticNet(n_actions=n_actions)

        self.actor_critic.compile(optimizer=optimizers.Adam(learning_rate=self.alpha))

    def choose_action(self, observation):

        state = tf.convert_to_tensor([observation])
        _, probs = self.actor_critic(state)

        action_probabilities = tfp.distributions.Categorical(probs = probs)
        action = action_probabilities.sample()
        self.action = action

        return action.numpy()[0]

    def save_models(self):

        print('saving models ....')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_models(self):

        print('... loading model')
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)

    def learn(self , state, reward, state_, done):

        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor([reward], dtype=tf.float32)

        with tf.GradientTape() as tape:

            state_value ,probs = self.actor_critic(state)# current state
            state_value_ , _ = self.actor_critic(state_)# new state
            state_value = tf.squeeze(state_value) #rende vettore 1x1 uno scalare
            state_value_ = tf.squeeze(state_value_)

            action_probs = tfp.distributions.Categorical(probs = probs)
            log_prob = action_probs.log_prob(self.action)

            delta = reward + self.gamma*state_value_*(1-int(done)) - state_value
            actor_loss = -log_prob*delta
            critic_loss = delta**2

            total_loss =actor_loss+critic_loss

        gradient = tape.gradient(total_loss,self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(
            gradient,self.actor_critic.trainable_variables
        ))








