import os
import pickle
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt


def plot_error(errors, title, pos):
    plt.figure(pos)
    plt.clf()
    plt.title(title)
    plt.xlabel('# of Iterations')
    plt.ylabel('Q-value Difference')
    plt.ylim(0, 0.5)
    plt.plot(errors, linestyle='-', color='black', linewidth=0.3)
    plt.pause(0.001)


def error_plot(errors, title):
    plt.plot(errors, linestyle='-', linewidth=0.6)
    plt.title(title)
    # plt.ylim(0, 0.5)
    plt.xlabel('Simulation Iteration')
    plt.ylabel('Q-value Difference')
    plt.ticklabel_format(style='sci', axis='x',
                         scilimits=(0, 0), useMathText=True)
    plt.show()

    fig_path = os.path.join('checkpoint', title)
    now = datetime.now()

    # dd_mm_YY_H_M_S
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    plt.savefig(fig_path + "_" + dt_string + ".png")


def save_obj(obj, name):
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    with open('checkpoint/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('checkpoint/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# Take action with epsilon-greedy r
def take_epsilon_greedy_action(Q, state_, epsilon_):
    # epsilon-greedy to take best action from action-value function
    if np.random.random() < epsilon_:
        return np.random.choice([0, 1, 2, 3, 4], 1)[0]

    max_idx = np.where(Q[state_[0]][state_[1]][state_[2]] == np.max(Q[state_[0]][state_[1]][state_[2]]))
    return max_idx[1][np.random.choice(range(len(max_idx[0])), 1)[0]]
