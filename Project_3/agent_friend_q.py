import time
from datetime import datetime

import numpy as np

from env import Soccer
from utils import take_epsilon_greedy_action, save_obj, error_plot


def Friend_Q(no_steps=int(1e6)):
    np.random.seed(1234)

    # discount factor
    gamma = 0.9

    # Define the epsilon and its decay for epsilon-greedy action selection
    epsilon = 1.0
    epsilon_min = 0.001
    epsilon_decay = 0.999995

    # learning rate
    alpha = 1.0
    alpha_min = 0.001

    # store the step-error
    error_list = []

    # Q_tables of player A and player B
    # the state-action space is 8 (pos for player A) * 8 (pos for player B) * 2 (ball possession)
    # * 5 (valid actions for player A) * 5 (valid actions for player B)
    # We encode 4*4=8 positions as linear instead of 2 dimensional matrix
    Q_1 = np.zeros((8, 8, 2, 5, 5))
    Q_2 = np.zeros((8, 8, 2, 5, 5))

    # index for step
    i = 0

    start_time = time.time()

    while i < no_steps:
        env = Soccer()

        # map two players positions and ball possession into state presentation
        state = [env.pos[0][0] * 4 + env.pos[0][1], env.pos[1][0] * 4 + env.pos[1][1], env.ball]

        while True:
            if i % 10 == 0:
                print('\rstep {}\t Time: {:.2f} \t Percentage: {:.2f}% \t Alpha: {:.3f}'.format(i,
                                                                                                time.time() - start_time,
                                                                                                i * 100 / no_steps,
                                                                                                alpha), end="")

            # player A at sate S take action South before update
            # first index is player A's position index (0-7), 2 is first row (0), 3rd column
            # second index is player B's position index (0-7), 1 is first row (0), 2nd column
            # third index is ball possession, according to graph, B has the ball
            # fourth index is action from player B, B sticks
            # fifth index is action from player A, A goes south
            # rationale for putting player A's action as last index is for easy handling of max
            # function (put the last dimension as player's action rather than opponent's action)
            before = Q_1[2][1][1][4][2]

            actions = [take_epsilon_greedy_action(Q_1, state, epsilon), take_epsilon_greedy_action(Q_2, state, epsilon)]

            # get next state, reward and game termination flag
            state_prime, rewards, done = env.move(actions)

            i += 1

            # Friend-Q is Q-learning adding dimension of opponent's action
            # state[0] = player1 state which is encoded as 1 dimensional value instead of 2 dimensional by using
            # mapping function = (posX * 4) + posY
            # state[1] = player1 state which is encoded as 1 dimensional value instead of 2 dimensional by using
            # mapping function = (posX * 4) + posY
            # state[3] = 0 if player1 has ball and 1 if player 2 has ball procession
            Q_1[state[0]][state[1]][state[2]][actions[1]][actions[0]] = \
                Q_1[state[0]][state[1]][state[2]][actions[1]][actions[0]] + alpha * (
                        rewards[0] + gamma * np.max(Q_1[state_prime[0]][state_prime[1]][state_prime[2]]) -
                        Q_1[state[0]][state[1]][state[2]][actions[1]][actions[0]])

            Q_2[state[0]][state[1]][state[2]][actions[0]][actions[1]] = \
                Q_2[state[0]][state[1]][state[2]][actions[0]][actions[1]] + alpha * (
                        rewards[1] + gamma * np.max(Q_2[state_prime[0]][state_prime[1]][state_prime[2]]) -
                        Q_2[state[0]][state[1]][state[2]][actions[0]][actions[1]])
            state = state_prime

            # player A at state S take action South before update
            after = Q_1[2][1][1][4][2]
            error_list.append(abs(after - before))

            if done:
                break

            # decay epsilon
            epsilon *= epsilon_decay
            epsilon = max(epsilon_min, epsilon)
            # decay alpha
            alpha = 1 / (i / alpha_min / no_steps + 1)

    d = {
        "error_list": error_list,
        "Q_1": Q_1,
        "Q_2": Q_2,
    }
    now = datetime.now()
    # dd_mm_YY_H_M_S
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    save_obj(d, "Friend_Q_Learning_" + dt_string)

    return error_list, Q_1, Q_2


if __name__ == '__main__':
    # Friend-Q
    friend_q_errors, Q_1_friend, Q_2_friend = Friend_Q()

    # error_plot(np.array(friend_q_errors), 'Friend_Q_Learning')

    # based on Taka's comment in office hours, just dropping the zeros from plot.
    # Frankly I don't quite understand why are we dropping zeros. Maybe thats what paper did and to replicate it we
    # need to drop as well
    error_plot(np.array(friend_q_errors)[np.where(np.array(friend_q_errors) > 0)], 'Friend_Q_Learning')
