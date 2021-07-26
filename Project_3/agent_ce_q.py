import time
from datetime import datetime

import numpy as np
import scipy
from cvxopt import solvers, matrix
from scipy.linalg import block_diag

from env import Soccer
from utils import save_obj, error_plot


def CE_Q(no_steps=int(1e6)):
    # Take action with epsilon-greedy
    def take_action(Pi_, state_, i_):
        # decay epsilon
        epsilon = epsilon_decay ** i_

        if np.random.random() < epsilon:
            index = np.random.choice(np.arange(25), 1)
            return np.array([index // 5, index % 5]).reshape(2)

        else:
            index = np.random.choice(np.arange(25), 1, p=Pi_[state_[0]][state_[1]][state_[2]].reshape(25))
            return np.array([index // 5, index % 5]).reshape(2)

    # using LP to solve correlated-equilibrium
    def solve_ce(Q_1_, Q_2_, state_):
        # For given state i.e. [Player_1_state, Player_2_state, ball_possession]: get the action matrix [5,5].
        Q_states_player_1 = Q_1_[state_[0]][state_[1]][state_[2]]

        s = scipy.linalg.block_diag(Q_states_player_1 - Q_states_player_1[0, :],
                                    Q_states_player_1 - Q_states_player_1[1, :],
                                    Q_states_player_1 - Q_states_player_1[2, :],
                                    Q_states_player_1 - Q_states_player_1[3, :],
                                    Q_states_player_1 - Q_states_player_1[4, :])
        row_index = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23]
        parameters_1 = s[row_index, :]

        # For given state i.e. [Player_1_state, Player_2_state, ball_possession]: get the action matrix [5,5].
        Q_states_player_2 = Q_2_[state_[0]][state_[1]][state_[2]]
        s = scipy.linalg.block_diag(Q_states_player_2 - Q_states_player_2[0, :],
                                    Q_states_player_2 - Q_states_player_2[1, :],
                                    Q_states_player_2 - Q_states_player_2[2, :],
                                    Q_states_player_2 - Q_states_player_2[3, :],
                                    Q_states_player_2 - Q_states_player_2[4, :])
        col_index = [0, 5, 10, 15, 20, 1, 6, 11, 16, 21, 2, 7, 12, 17, 22, 3, 8, 13, 18, 23, 4, 9, 14, 19, 24]
        parameters_2 = s[row_index, :][:, col_index]

        c = matrix((Q_1_[state_[0]][state_[1]][state_[2]] + Q_2_[state_[0]][state_[1]][state_[2]].T).reshape(25))
        # construct probability constraints
        A = matrix(np.ones((1, 25)))
        b = matrix(1.0)
        # construct rationality constraints
        G = matrix(np.append(np.append(parameters_1, parameters_2, axis=0), -np.eye(25), axis=0))
        h = matrix(np.zeros(65) * 0.0)

        # error-handling mechanism
        try:
            sol = solvers.lp(c=c, G=G, h=h, A=A, b=b)
            if sol['x'] is not None:
                prob_ = np.abs(np.array(sol['x']).reshape((5, 5))) / sum(np.abs(sol['x']))
                val_1_ = np.sum(prob_ * Q_1_[state_[0]][state_[1]][state_[2]])
                val_2_ = np.sum(prob_ * Q_2_[state_[0]][state_[1]][state_[2]].T)
            else:
                prob_ = None
                val_1_ = None
                val_2_ = None
        except:
            print("error!!")
            prob_ = None
            val_1_ = None
            val_2_ = None

        return prob_, val_1_, val_2_

    # discount rate
    gamma = 0.9
    epsilon_min = 0.001
    epsilon_decay = 10 ** (np.log10(epsilon_min) / no_steps)
    # epsilon_min = 0.001
    # epsilon_decay = 0.999995

    # learning rate
    alpha = 1
    alpha_min = 0.001
    alpha_decay = 10 ** (np.log10(alpha_min) / no_steps)

    # Q_tables of player A and player B
    # the state-action space is 8 (pos for player A) * 8 (pos for player B) * 2 (ball possession) *
    # 5 (valid actions for player A) * 5 (valid actions for player B)
    # We encode 4*4=8 positions as linear instead of 2 dimensional matrix
    Q_1 = np.ones((8, 8, 2, 5, 5)) * 1.0
    Q_2 = np.ones((8, 8, 2, 5, 5)) * 1.0

    # value of states, only depends on pos of players and possession of ball
    V_1 = np.ones((8, 8, 2)) * 1.0
    V_2 = np.ones((8, 8, 2)) * 1.0

    # shared joint policy
    Pi = np.ones((8, 8, 2, 5, 5)) * 1 / 25

    # error list to store ERR
    error_list = []

    # set seed
    np.random.seed(1234)

    start_time = time.time()
    i = 0
    while i < no_steps:
        soccer = Soccer()
        state = [soccer.pos[0][0] * 4 + soccer.pos[0][1], soccer.pos[1][0] * 4 + soccer.pos[1][1], soccer.ball]
        done = 0
        j = 0
        while not done and j <= 100:

            if i % 1000 == 0:
                print('\rstep {}\t Time: {:.2f} \t Percentage: {:.2f}% \t Alpha: {:.3f}'.format(i,
                                                                                                time.time() - start_time,
                                                                                                i * 100 / no_steps,
                                                                                                alpha), end="")
            # update index
            i, j = i + 1, j + 1

            # we don't need place player B action space before player A
            # since we are no longer just selecting the max of player A
            before = Q_1[2][1][1][2][4]

            # eps-greedy to generate action
            actions = take_action(Pi, state, i)

            state_prime, rewards, done = soccer.move(actions)

            # Q-learning update
            # state[0] = player1 state which is encoded as 1 dimensional value instead of 2 dimensional by using
            # mapping function = (posX * 4) + posY
            # state[1] = player1 state which is encoded as 1 dimensional value instead of 2 dimensional by using
            # mapping function = (posX * 4) + posY
            # state[3] = 0 if player1 has ball and 1 if player 2 has ball procession
            Q_1[state[0]][state[1]][state[2]][actions[0]][actions[1]] = (1 - alpha) * \
                                                                        Q_1[state[0]][state[1]][state[2]][actions[0]][
                                                                            actions[1]] + alpha * (rewards[0] + gamma *
                                                                                                   V_1[state_prime[0]][
                                                                                                       state_prime[1]][
                                                                                                       state_prime[2]])

            # Q-learning update
            Q_2[state[0]][state[1]][state[2]][actions[1]][actions[0]] = (1 - alpha) * \
                                                                        Q_2[state[0]][state[1]][state[2]][actions[1]][
                                                                            actions[0]] + alpha * (rewards[1] + gamma *
                                                                                                   V_2[state_prime[0]][
                                                                                                       state_prime[1]][
                                                                                                       state_prime[
                                                                                                           2]].T)
            prob, val_1, val_2 = solve_ce(Q_1, Q_2, state)

            # update only if not Null returned from the ce solver
            if prob is not None:
                Pi[state[0]][state[1]][state[2]] = prob
                V_1[state[0]][state[1]][state[2]] = val_1
                V_2[state[0]][state[1]][state[2]] = val_2
            state = state_prime

            # player A at state S take action South after update
            after = Q_1[2][1][1][2][4]
            # compute the error
            error_list.append(np.abs(after - before))

            # decay alpha
            alpha = alpha_decay ** i

    d = {
        "error_list": error_list, "Q_1": Q_1, "Q_2": Q_2, "V_1": V_1, "V_2": V_2, "Pi": Pi
    }
    now = datetime.now()
    # dd/mm/YY_H:M:S
    dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
    save_obj(d, "CE_Q_1_" + dt_string)

    return error_list, Q_1, Q_2, V_1, V_2, Pi


if __name__ == '__main__':
    # CE_Q
    ce_q_errors, Q_1_ce, Q_2_ce, V_1_ce, V_2_ce, Pi_ce = CE_Q()

    error_plot(np.array(ce_q_errors), 'CE-Q')
