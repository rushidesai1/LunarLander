import os
from datetime import datetime

import gym
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch

from Project_2.lunar_lander.environment import Environment
from Project_2.lunar_lander.q_agent import Q_Agent


class ExperimentParameters:
    def __init__(self,
                 experiment_name="best-exp",
                 num_episodes=500,
                 checkpoint_freq=100,
                 print_freq=1,
                 load_checkpoint=None,
                 timeout=1600
                 ):
        # Based on paper https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
        self.num_episodes = num_episodes
        self.print_freq = print_freq
        self.load_checkpoint = load_checkpoint
        self.max_steps_per_episode = timeout

        self.experiment_name = experiment_name
        self.checkpoint_freq = checkpoint_freq


def get_average(agent_sum_reward, num=50):
    agent_sum_reward = np.array(agent_sum_reward)
    if len(agent_sum_reward) < num:
        return agent_sum_reward.mean()
    else:
        return agent_sum_reward[-num:].mean()


def save_fig(agent_sum_reward, average_sum_reward, npy_path, fig_path, agent_name, env_name):
    np.save(npy_path, np.vstack([agent_sum_reward, average_sum_reward]))
    x = np.load(npy_path)
    plt.title("{} on {}".format(agent_name, env_name))
    plt.plot(np.arange(x.shape[1]), x[0], color='blue')
    plt.plot(np.arange(x.shape[1]), x[1], color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend(['Actual Reward', 'Running Average'])
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path)


def update_agent_parameters(environment_parameters, agent_parameters):
    env = gym.make(environment_parameters['gym_environment'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent_parameters['network_config']['state_dim'] = state_dim
    agent_parameters['network_config']['action_dim'] = action_dim

    checkpoint_dir = agent_parameters['checkpoint_dir']
    agent_parameters['checkpoint_dir'] = os.path.join(checkpoint_dir, environment_parameters['gym_environment'],
                                                      agent_parameters['name'])
    os.makedirs(agent_parameters['checkpoint_dir'], exist_ok=True)
    return agent_parameters


def run_experiment(environment_parameters, agent_parameters, experiment_parameters: ExperimentParameters):
    num_episodes = 0

    # save sum of reward at the end of each episode
    agent_sum_reward = []
    average_sum_reward = []

    agent_parameters = agent_parameters

    env = environment_parameters["env_instance"]
    environment: Environment = Environment(env)
    agent: Q_Agent = Q_Agent(agent_parameters)

    starting_episode = 0

    agent_name = agent_parameters['name']
    save_name = "{}.npy".format(agent.name)
    npy_path = os.path.join(agent.checkpoint_dir, "sum_reward_{}".format(save_name))
    fig_path = os.path.join(agent.checkpoint_dir, 'sum_rewards.png')

    # load checkpoint if any
    if experiment_parameters.load_checkpoint is not None:
        agent.load_checkpoint(experiment_parameters.load_checkpoint)
        agent_sum_reward, average_sum_reward = np.load(npy_path)
        agent_sum_reward = list(agent_sum_reward)
        average_sum_reward = list(average_sum_reward)
        fname = experiment_parameters.load_checkpoint.split(os.path.sep)[-1]
        try:
            starting_episode = int(fname.split('_')[1])
        except IndexError:
            starting_episode = len(agent_sum_reward)

        print(f"starting from episode {starting_episode}")

    # mlflow.create_experiment(experiment_parameters.experiment_name)
    mlflow.set_experiment(experiment_parameters.experiment_name)
    mlflow.pytorch.autolog()

    # mlflow.log_metric("training_time", )

    with mlflow.start_run():
        mlflow.log_param("agent_params", agent_parameters)
        mlflow.log_param("experiment_params", experiment_parameters)
        for episode in range(1 + starting_episode, experiment_parameters.num_episodes + 1):
            # ------------- run episode start----------------
            episode_total_reward = 0.0
            num_steps = 1
            reward = 0

            state = environment.reset()
            last_action = agent.agent_start(state)
            is_done = False

            while (not is_done) and ((experiment_parameters.max_steps_per_episode == 0) or
                                     (num_steps < experiment_parameters.max_steps_per_episode)):
                (reward, state, is_done) = environment.env_step(last_action)

                episode_total_reward += reward

                last_action = agent.agent_step(reward, state)
                num_steps += 1

            num_episodes += 1
            agent.agent_end(reward)
            # ------------- run episode end----------------

            agent_sum_reward.append(episode_total_reward)
            mlflow.log_metric("EpisodeReward", episode_total_reward, episode)
            # mlflow.log_metric("Reward", episode_reward, episode)
            if episode % experiment_parameters.print_freq == 0:
                print(
                    'Episode {}/{} | Reward {}'.format(episode, experiment_parameters.num_episodes,
                                                       episode_total_reward))

            average = get_average(agent_sum_reward)
            average_sum_reward.append(average)

            gym_name = environment_parameters['gym_environment']
            if episode % experiment_parameters.checkpoint_freq == 0:
                agent.save_checkpoint(episode)
                save_fig(agent_sum_reward, average_sum_reward, npy_path, fig_path, agent_name, gym_name)

            if environment_parameters['solved_threshold'] is not None \
                    and average >= environment_parameters['solved_threshold']:
                print("Task Solved with reward = {}".format(episode_total_reward))
                agent.save_checkpoint(episode, done=True)
                break

    save_fig(agent_sum_reward, average_sum_reward, npy_path, fig_path, agent_name,
             environment_parameters['gym_environment'])

    mlflow.log_artifacts('model_weights')


def main():
    now = datetime.now()

    # dd/mm/YY_H:M:S
    dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")

    # Experiment parameters
    experiment_parameters = ExperimentParameters(
        experiment_name="exp-tau-0.001-gamma-0.99_temperature_softmax" + dt_string,
        num_episodes=500,
        checkpoint_freq=100,
        print_freq=1,
        load_checkpoint=None,
        # OpenAI Gym environments allow for a time-step limit timeout, causing episodes to end after
        # some number of time-steps.
        timeout=1600
    )

    # Environment parameters
    environment_parameters = {
        "gym_environment": 'LunarLander-v2',
        'solved_threshold': 200,
        'seed': 0
    }
    env = gym.make(environment_parameters['gym_environment'])
    environment_parameters['env_instance'] = env

    # Agent parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent_parameters = {
        # Network has 3 layers
        # input, 1 hidden, output layer
        'network_config': {
            # comes from # of states
            'state_dim': 8,
            # tried 64, 128, 256 layers and 256 worked. So kept it.
            # Since 1 layer worked, didn't try adding more hidden layers.
            # (Also took inspiration from MNIST pytorch example)
            'num_hidden_units': 256,
            # comes from # of actions
            'action_dim': 4,
            'seed': 0
        },
        'optimizer_config': {
            'step_size': 1e-3,
            'betas': (0.9, 0.999)
        },
        'name': 'q-learning',
        'device': device,
        'replay_buffer_size': 50000,
        'mini_batch_size': 64,
        'num_replay_updates_per_step': 4,
        'gamma': 0.99,
        'tau': 0.001,
        'checkpoint_dir': 'model_weights',
        'seed': 0
    }

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent_parameters['network_config']['state_dim'] = state_dim
    agent_parameters['network_config']['action_dim'] = action_dim

    checkpoint_dir = agent_parameters['checkpoint_dir']
    agent_parameters['checkpoint_dir'] = os.path.join(checkpoint_dir, environment_parameters['gym_environment'],
                                                      agent_parameters['name'])
    os.makedirs(agent_parameters['checkpoint_dir'], exist_ok=True)

    # run experiment
    run_experiment(environment_parameters, agent_parameters, experiment_parameters)


# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory
if __name__ == '__main__':
    main()
