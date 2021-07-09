import argparse
from datetime import datetime
from unittest import TestCase

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from Project_2.lunar_lander.test import fetch_agent_env_from_checkpoint


class Test(TestCase):
    def test_agent_rewards(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--env", type=str, default="LunarLander-v2", help="Environment name")
        parser.add_argument("--agent", type=str, default="q-learning", help="Agent name")
        parser.add_argument("--checkpoint", type=str, help="Name of checkpoint.pth file under model_weights/env/")
        parser.add_argument("--gif", action='store_true',
                            help='Save rendered episode as a gif to model_weights/env/recording.gif')
        opt = parser.parse_args()

        agent, checkpoint_dir, env = fetch_agent_env_from_checkpoint(opt)

        now = datetime.now()
        # dd/mm/YY_H:M:S
        dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
        mlflow.set_experiment("test_experiment-" + dt_string)

        rewards_100_episodes = []
        for i in range(100):
            last_state = env.reset()
            action = agent.policy(last_state)
            done = False
            episode_rewards = []
            while not done:
                state, reward, done, info = env.step(action)
                action = agent.policy(state)
                episode_rewards.append(reward)
            avg_episode_reward = np.average(np.array(episode_rewards))
            rewards_100_episodes.append(avg_episode_reward)
            mlflow.log_metric("reward_episode", avg_episode_reward, i)

        plt.plot(rewards_100_episodes)
        plt.show()
