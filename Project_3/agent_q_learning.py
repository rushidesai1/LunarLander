import time
from datetime import datetime

import numpy as np

from env import Soccer
from utils import save_obj, error_plot


def Q_learning(no_steps=int(1e6)):
    # Take action with epsilon-greedy r
    def generate_action(Q, state_, epsilon_):
        # epsilon-greedy to take best action from action-value function
        if np.random.random() < epsilon_:
            return np.random.choice([0, 1, 2, 3, 4], 1)[0]

        return np.random.choice(
            np.where(Q[state_[0]][state_[1]][state_[2]] == max(Q[state_[0]][state_[1]][state_[2]]))[0], 1
        )[0]

    np.random.seed(1)

    # discount factor
    gamma = 0.9

    # Define the epsilon and its decay for epsilon-greedy action selection
    epsilon = 1.0
    epsilon_min = 0.001
    epsilon_decay = 0.999995

    # learning rate
    alpha = 1.0
    alpha_min = 0.001
    alpha_decay = 0.999995

    # store the step-error
    error_list = []

    # Q_tables of player A and player B
    # the state-action space is 8 (pos for player A) * 8 (pos for player B) * 2 (ball possession) * 5 (valid actions)
    Q_1 = np.zeros((8, 8, 2, 5))
    Q_2 = np.zeros((8, 8, 2, 5))

    # index for step
    i = 0

    start_time = time.time()

    while i < no_steps:
        env = Soccer()

        # map two players positions and ball possession into state presentation
        state = [env.pos[0][0] * 4 + env.pos[0][1], env.pos[1][0] * 4 + env.pos[1][1], env.ball]

        while True:
            if i % 100 == 0:
                print('\rstep {}\t Time: {:.2f} \t Percentage: {:.2f}% \t Alpha: {:.3f}'.format(i,
                                                                                                time.time() - start_time,
                                                                                                i * 100 / no_steps,
                                                                                                alpha), end="")

            # player A at sate S take action South before update
            # first index is player A's position index (0-7), 2 is first row (0), 3rd column
            # second index is player B's position index (0-7), 1 is first row (0), 2nd column
            # third index is ball possession, according to graph, B has the ball
            # fourth index is action from player A, A is going south (2)
            before = Q_1[2][1][1][2]

            # eps-greedy to generate action
            actions = [generate_action(Q_1, state, epsilon), generate_action(Q_2, state, epsilon)]
            # get next state, reward and game termination flag
            state_prime, rewards, done = env.move(actions)

            i += 1

            # Q-learning for player A & B
            # state[0] = player1 state which is encoded as 1 dimensional value instead of 2 dimensional by using
            # mapping function = (posX * 4) + posY
            # state[1] = player1 state which is encoded as 1 dimensional value instead of 2 dimensional by using
            # mapping function = (posX * 4) + posY
            # state[3] = 0 if player1 has ball and 1 if player 2 has ball procession
            Q_1[state[0]][state[1]][state[2]][actions[0]] = Q_1[state[0]][state[1]][state[2]][
                                                                actions[0]] + alpha * (rewards[0] + gamma * max(
                Q_1[state_prime[0]][state_prime[1]][state_prime[2]]) - Q_1[state[0]][state[1]][state[2]][
                                                                                           actions[0]])

            Q_2[state[0]][state[1]][state[2]][actions[1]] = Q_2[state[0]][state[1]][state[2]][
                                                                actions[1]] + alpha * (rewards[1] + gamma * max(
                Q_2[state_prime[0]][state_prime[1]][state_prime[2]]) - Q_2[state[0]][state[1]][state[2]][
                                                                                           actions[1]])
            state = state_prime

            # player A at state S take action South before update
            after = Q_1[2][1][1][2]
            error_list.append(abs(after - before))

            if done:
                break

            # decay epsilon and alpha
            epsilon *= epsilon_decay
            epsilon = max(epsilon_min, epsilon)

            alpha *= alpha_decay
            alpha = max(alpha_min, alpha)

    d = {
        "error_list": error_list,
        "Q_1": Q_1,
        "Q_2": Q_2,
    }
    now = datetime.now()
    # dd_mm_YY_H_M_S
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    save_obj(d, "Q_learning_" + dt_string)

    return error_list, Q_1, Q_2


if __name__ == '__main__':
    # Q-learning
    q_learning_errors, Q_1_q_learning, Q_2_q_learning = Q_learning()

    error_plot(np.array(q_learning_errors), 'Q_Learning')
