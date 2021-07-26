# Foe-Q Learning
import time
from datetime import datetime

import numpy as np
from cvxopt import matrix, solvers

from env import Soccer
from utils import save_obj, error_plot


def Foe_Q(no_steps=int(1e6)):
    # Take action with epsilon-greedy
    def generate_action(pi, state, i):
        # epsilon-greedy to take best action from action-value function
        # decay epsilon
        epsilon = epsilon_decay ** i
        if np.random.random() < epsilon:
            return np.random.choice([0, 1, 2, 3, 4], 1)[0]
        else:
            return np.random.choice([0, 1, 2, 3, 4], 1, p=pi[state[0]][state[1]][state[2]])[0]

    # same formulation as hw6
    # Q value is just like the reward matrix
    def max_min(Q, state_):
        c = matrix([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        G = matrix(np.append(np.append(np.ones((5, 1)), -Q[state_[0]][state_[1]][state_[2]], axis=1),
                             np.append(np.zeros((5, 1)), -np.eye(5), axis=1), axis=0))
        h = matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        A = matrix([[0.0], [1.0], [1.0], [1.0], [1.0], [1.0]])
        b = matrix(1.0)
        sol = solvers.lp(c=c, G=G, h=h, A=A, b=b)
        return np.abs(sol['x'][1:]).reshape((5,)) / sum(np.abs(sol['x'][1:])), np.array(sol['x'][0])

    # discount factor
    gamma = 0.9

    # Define the epsilon and its decay for epsilon-greedy action selection
    epsilon_min = 0.001
    epsilon_decay = 10 ** (np.log10(epsilon_min) / no_steps)

    # learning rate
    alpha_min = 0.001
    # from litman
    alpha_decay = 10 ** (np.log10(alpha_min) / no_steps)
    alpha = alpha_decay ** 0

    # Q_tables of player A and player B
    # the state-action space is 8 (pos for player A) * 8 (pos for player B) * 2 (ball possession)
    # * 5 (valid actions for player A) * 5 (valid actions for player B)
    # initialization to 1 in order to break from zero
    Q_1 = np.ones((8, 8, 2, 5, 5)) * 1.0
    Q_2 = np.ones((8, 8, 2, 5, 5)) * 1.0

    # init policy for player 1 and player 2
    Pi_1 = np.ones((8, 8, 2, 5)) * 1 / 5
    Pi_2 = np.ones((8, 8, 2, 5)) * 1 / 5

    # value of states, only depends on pos of players and possession of ball
    V_1 = np.ones((8, 8, 2)) * 1.0
    V_2 = np.ones((8, 8, 2)) * 1.0

    # error list to store ERR
    errors_list = []

    # set seed
    np.random.seed(1234)

    # Loop for no_steps steps
    start_time = time.time()
    i = 0

    while i < no_steps:
        soccer = Soccer()
        state = [soccer.pos[0][0] * 4 + soccer.pos[0][1], soccer.pos[1][0] * 4 + soccer.pos[1][1], soccer.ball]
        done = 0
        while not done:
            if i % 1000 == 0:
                print('\rstep {}\t Time: {:.2f} \t Percentage: {:.2f}% \t Alpha: {:.3f}'.format(i,
                                                                                                time.time() - start_time,
                                                                                                i * 100 / no_steps,
                                                                                                alpha), end="")
            i += 1

            # player A at sate S take action South before update
            # first index is player A's position index (0-7), 2 is frist row (0), 3rd column
            # second index is player B's position index (0-7), 1 is first row (0), 2nd column
            # third index is ball possession, according to graph, B has the ball
            # fourth index is action from player B, B sticks
            # fifth index is action from player A, A goes south
            # rationale for putting player A's action as last index is for easy handling of max
            # function (put the last dimension as player's action rather than opponent's action)
            before = Q_1[2][1][1][4][2]

            # eps-greedy to generate action
            actions = [generate_action(Pi_1, state, i), generate_action(Pi_2, state, i)]

            # get next state, reward and game termination flag
            state_prime, rewards, done = soccer.move(actions)

            # Foe-Q-learning update
            # state[0] = player1 state which is encoded as 1 dimensional value instead of 2 dimensional by using
            # mapping function = (posX * 4) + posY
            # state[1] = player1 state which is encoded as 1 dimensional value instead of 2 dimensional by using
            # mapping function = (posX * 4) + posY
            # state[3] = 0 if player1 has ball and 1 if player 2 has ball procession
            Q_1[state[0]][state[1]][state[2]][actions[1]][actions[0]] = (1 - alpha) * \
                                                                        Q_1[state[0]][state[1]][state[2]][actions[1]][
                                                                            actions[0]] + alpha * (rewards[0] + gamma *
                                                                                                   V_1[state_prime[0]][
                                                                                                       state_prime[1]][
                                                                                                       state_prime[2]])

            # use LP to solve max-min
            pi, val = max_min(Q_1, state)
            Pi_1[state[0]][state[1]][state[2]] = pi
            V_1[state[0]][state[1]][state[2]] = val

            # Q-learning update
            Q_2[state[0]][state[1]][state[2]][actions[0]][actions[1]] = (1 - alpha) * \
                                                                        Q_2[state[0]][state[1]][state[2]][actions[0]][
                                                                            actions[1]] + alpha * (rewards[1] + gamma *
                                                                                                   V_2[state_prime[0]][
                                                                                                       state_prime[1]][
                                                                                                       state_prime[2]])

            # use LP to solve max-min
            pi, val = max_min(Q_2, state)
            Pi_2[state[0]][state[1]][state[2]] = pi
            V_2[state[0]][state[1]][state[2]] = val
            state = state_prime

            # compute ERR
            after = Q_1[2][1][1][4][2]
            errors_list.append(np.abs(after - before))

            # decay learning rate
            alpha = alpha_decay ** i

    d = {
        "errors_list": errors_list, "Q_1": Q_1, "Q_2": Q_2, "V_1": V_1, "V_2": V_2, "Pi_1": Pi_1, "Pi_2": Pi_2,
    }
    now = datetime.now()
    # dd_mm_YY_H_M_S
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    save_obj(d, "Foe_Q_" + dt_string)

    return errors_list, Q_1, Q_2, V_1, V_2, Pi_1, Pi_2


if __name__ == '__main__':
    # Foe_Q
    foe_q_errors, Q_1, Q_2, V_1, V_2, Pi_1, Pi_2 = Foe_Q()

    error_plot(np.array(foe_q_errors), 'Foe-Q')
