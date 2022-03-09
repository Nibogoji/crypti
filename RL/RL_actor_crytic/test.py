import gym
import numpy as np
from RL.RL_actor_crytic import Agent
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

if __name__=='__main__':

    env = gym.make('CartPole-v0')
    agent = Agent(alpha=1e-5, n_actions= env.action_space.n)
    n_games  = 1800

    filename = 'cartpole.png'
    figure_file = 'plots/'+filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:

        agent.load_models()

    for i in range(n_games):

        observation = env.reset()
        done = False
        score = 0
        while not done:

            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:

                agent.learn(observation, reward, observation_, done)
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score>best_score:

            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print('episoode ', i, 'score %.1f'% score,'avg_score %.1f'%avg_score)

    x = [i for i in range(n_games)]
    plot_learning_curve(x,score_history, figure_file)
