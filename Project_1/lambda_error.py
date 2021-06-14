import pickle

import numpy as np
from matplotlib import pyplot as plt

actual_values = np.array([1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6])

training_set_size = 10
training_sequences = 100
# iterations = 20
# Ran binary search (kind of) to find alpha
# 0.1 didnt converge
# 0.01 didnt converge
# 0.05 didnt converge
# 0.005 didnt converge
# 0.001 converged 0.03 to 0.006
# 0.0004 slower converge 0.03 to 0.06 than 0.001
# 0.0005 slower converge 0.03 to 0.06 than 0.004
# 0.0001 slower converge 0.03 to 0.06 than 0.005
alpha = 0.001


# # Used to generate data, basically takes the random walk as described in Sutton paper
def random_walks(num_steps, start_step, seq_per_set, num_sets, seed=-1) -> list[list]:
    """
    Create a list of lists of training sequences for random walks.
    :param num_steps: The number of steps in the random walk
    :param start_step: The starting step of the sequences. -1 for random
    :param seq_per_set: Number of training sequences in each training set
    :param num_sets: Number of training sets
    :param seed: The random seed to use for generating steps. Use -1 for no seed
    :return training: Training data. Access a sequence (matrix) with training[set][seq]
    """
    # Set the random seed, if supplied
    if seed > 0:
        np.random.seed(seed)
    # init training_set data-structure
    training_set: list[list] = num_sets * [seq_per_set * [None]]
    # Iterate to build the training data randomly
    for set1 in range(num_sets):  # for each set
        for seq in range(seq_per_set):  # for each sequence
            if start_step == -1:  # pick start location randomly
                start_step = np.random.randint(1, num_steps)
            # Initialize the sequence
            step = start_step
            sequence = np.zeros(num_steps).astype(int)
            sequence[step] = 1
            while is_not_absorbing_state(num_steps, step):  # while not in absorbing state
                if np.random.uniform() >= 0.5:  # Uniformly random L v R step
                    step += 1  # Go right
                else:
                    step -= 1  # Go left
                # vector representing this sequence
                training_sequence = np.zeros(num_steps).astype(int)
                # Set the appropriate element to 1
                training_sequence[step] = 1
                # Add this step to the sequence
                sequence = np.vstack((sequence, training_sequence))
            # Assign the sequence to its position in the training data
            training_set[set1][seq] = sequence
    return training_set


def is_not_absorbing_state(num_steps, step):
    return step != 0 and step != num_steps - 1


# -----------------------------------------------------------------------------------------------
def learn_weight_updates(training_seqs, lambda_val, alpha, z_vals, w):
    """
    Given a set of x data, perform repeated weight updates as in eq 4 in Sutton_1988)
    When prediction P is linear combination of w and x i.e. P_t = w⊤ * x_t
            # ∆w = α * (z - w⊤ * x_t)x_t                                 (page 14 Sutton_1988)
            # To compute incrementally,
            # ∆w_t = α * (P_t+1 - P_t) * ∑ k=1..t ( ∇w * P_k )           (eq_3, page 15 Sutton_1988)
            # λ weighting is achieved by introducing λ in above eq
            # ∆w_t = α * (P_t+1 - P_t) * ∑ k=1..t λ^t-k * ( ∇w * P_k )   (eq_4, page 15 Sutton_1988)
            # Pt = w⊤ * x_t                                              (page 19 sutton_1988)

    :param training_seqs: The input x sequence
    :param lambda_val: ...lambda.
    :param alpha: Learning rate
    :param z_vals: A tuple of the form (r for state 0, r for state[-1])
    :param w: The weights coming in
    :return delta_p: A NumPy vector of weight values
    """
    # Determine the number of steps taken in the sequence, by the number of rows
    num_train_size = training_seqs.shape[0] - 1
    # Number of non-terminal states
    num_states = training_seqs.shape[1] - 2
    # Get the reward value
    z = z_vals[training_seqs[-1, -1]]
    # Initialize the lambda sequence and weight updates
    lambda_seq = np.ones(1)
    ones_list = np.ones(1)
    delta_w = np.zeros(num_states)
    # Chop off the reward t data
    x = training_seqs[:-1, 1:-1]
    # Perform the weight updates
    for t in range(num_train_size):
        prev_steps = x[0:t + 1]
        # print('w =', w)
        # print('Training sequence:')
        # print(prev_steps)
        # print('Lambda sequence:')
        # print(lambda_seq)
        # print('Lambda sequence * x sequence:')
        # print(np.sum(prev_steps * lambda_seq[:, None], axis=0))
        if t == num_train_size - 1:  # The t entering the absorbing state
            # print("z =", z)
            delta_p = z - np.dot(w, x[-1, :])
        else:  # Non-terminal state
            # P_t = w⊤ * x_t = Σi w(i)*x(i)         (when prediction is linear function of x and w page 13 sutton_1988)
            # P_t+1 = w⊤ * x_t+1                                              (above eq for t+1 step)
            delta_p = np.dot(w, x[t + 1, :]) - np.dot(w, x[t, :])
        # print('delta_p =', delta_p)

        # To compute incrementally,
        # ∆w_t = α * (P_t+1 - P_t) * ∑ k=1..t ( ∇w * P_k )           (eq_3, page 15 sutton_1988)
        # In above eq: delta_p = P_t+1 - P_t
        # λ weighting is achieved by introducing λ in above eq
        # ∆w_t = α * (P_t+1 - P_t) * ∑ k=1..t λ^t-k * ( ∇w * P_k )   (eq_4, page 15 sutton_1988)
        delta_w += alpha * delta_p * np.sum(prev_steps * lambda_seq[:, None], axis=0)

        # print('delta_p =', delta_p)
        lambda_seq = np.concatenate((lambda_seq * lambda_val, ones_list))
    return delta_w


# use_local = True
use_local = False

rmse_file_name = "RMSE_lambda_error_alpha_001_seed_33"

if use_local:
    a_file = open("%s.pkl" % rmse_file_name, "rb")
    RMSE_vector = pickle.load(a_file)
else:
    # Generate a random walk
    generated_training_set = random_walks(7, 3, training_sequences, training_set_size, seed=33)

    # Setup initial RMSE vector
    RMSE_vector = np.zeros(100 + 1)
    # lambda_vector = [0, 10, 30, 50, 70, 100]
    # lambda_vector = [0, 5, 10, 15, 20, 25, 30, 35, 40, 55, 50, 55, 65, 70, 75, 80, 85, 90, 95, 100]
    # for lambda_it in lambda_vector:
    for lambda_it in range(len(RMSE_vector)):
        # Reset weights and deltas
        weights = 0.5 * np.ones(5)
        new_weights = 0.5 * np.ones(5)
        deltas = np.zeros(5)

        # for iteration in range(iterations):
        while True:
            max_diff = float('-inf')
            for training_set in range(training_set_size):
                for training_seq in range(training_sequences):
                    # accumulate deltas
                    deltas += learn_weight_updates(training_seqs=generated_training_set[training_set][training_seq],
                                                   lambda_val=lambda_it / 100,
                                                   alpha=alpha, z_vals=(0, 1), w=weights)

                # single pass of training set completed, now update the weights
                weights += deltas
                max_diff = max(max_diff, abs(np.max(deltas)))
                deltas = np.zeros(5)

            # check for convergence
            if max_diff < 1e-4:
                break

        # w = [0.17628762 0.34308179 0.47018928 0.59858362 0.79113044]
        # 0.03871087807906376
        RMSE_vector[lambda_it] = np.sqrt(((weights - actual_values) ** 2).mean())
        print(str(lambda_it) + '% done')

    filehandler = open("%s.pkl" % rmse_file_name, 'wb')
    pickle.dump(RMSE_vector, filehandler)
    filehandler.close()

# Plot RMSE vs lambda
plt.plot(RMSE_vector)
plt.ylabel('RMSE')
plt.xlabel('λ')

plt.yticks(np.linspace(min(RMSE_vector), max(RMSE_vector), num=10))
# plt.xticks([0, 5, 10, 20, 30, 40, 60, 80, 100], ['0.0', '0.05', '0.1', '0.2', '0.3', '0.4', '0.6', '0.8', '1.0'])
plt.xticks([0, 10, 30, 50, 70, 100], ['0.0', '0.1', '0.3', '0.5', '0.7', '1.0'])
plt.title('Replication of Figure 3 in Sutton (1988)')

# Show the plot
plt.show()

# from scipy.interpolate import interp1d

# x = np.linspace(0, 10, num=11, endpoint=True)
# y = np.cos(-x ** 2 / 9.0)
# x = RMSE_vector
# y = range(0, 101)
# f = interp1d(x, y)
# f2 = interp1d(x, y, kind='cubic')
#
# xnew = np.linspace(0, 1, endpoint=True)
# plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
# plt.legend(['data', 'linear', 'cubic'], loc='best')
# plt.show()
#
