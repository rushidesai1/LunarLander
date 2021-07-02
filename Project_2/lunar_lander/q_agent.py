import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim

from .q_network import DQN
from .replay_buffer import ReplayBuffer


class Q_Agent(pl.LightningModule):
    def __init__(self, agent_config):
        """
        Lightening module helps with mlflow auto-log.
        Setup for the agent called when the experiment first starts.
        Assume agent_config dict contains:
        {
            network_config: dictionary,
            optimizer_config: dictionary,
            replay_buffer_size: integer,
            mini_batch_size: integer,
            num_replay_updates_per_step: float
            discount_factor: float,
        }
        """
        super().__init__()
        self._name = agent_config['name']
        self._device = agent_config['device']
        self._replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'],
                                           agent_config['mini_batch_size'],
                                           agent_config.get('seed'))
        self._optim_config = agent_config['optimizer_config']

        # The latest state of the network that is getting replay updates
        self._q_net = DQN(agent_config['network_config']).to(self._device)
        self.target_q_net = DQN(agent_config['network_config']).to(self._device)

        self._optimizer = self.configure_optimizers()
        self._num_actions = agent_config['network_config']['action_dim']
        self._num_replay = agent_config['num_replay_updates_per_step']
        self._discount = agent_config['gamma']
        self._tau = agent_config['tau']

        self._rand_generator = np.random.RandomState(agent_config.get('seed'))

        self._last_state = None
        self._last_action = None

        self._sum_rewards = 0
        self._episode_steps = 0

        checkpoint_dir = agent_config.get('checkpoint_dir')
        if checkpoint_dir is None:
            self.checkpoint_dir = 'model_weights'
        else:
            self.checkpoint_dir = checkpoint_dir

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def configure_optimizers(self):
        # return optim.RMSprop(self._q_net.parameters(), lr=self._optim_config['step_size'])
        return optim.Adam(self._q_net.parameters(), lr=self._optim_config['step_size'],
                          betas=self._optim_config['betas'])

    def training_step(self, experiences, batch_num: int):
        """
        network: The latest state of the network that is getting replay updates.
        network_target: The fixed network used for computing the targets,

        :param batch_num: just for adhering to mlflow contract
        :param experiences : The batch of experiences including the states, actions, rewards, terminals, and next_states
                             and particularly, the action-values at the next-states.
        """

        self.set_train()
        # Get states, action, rewards, terminals, and next_states from experiences
        states, actions, rewards, terminals, next_states = experiences

        # move tensors to appropriate device (i.e to cpu or gpu)
        states = torch.tensor(states).to(self._device).float()
        next_states = torch.tensor(next_states).to(self._device).float()
        actions = torch.tensor(actions).to(self._device)
        rewards = torch.tensor(rewards).to(self._device).float()
        terminals = torch.tensor(terminals).to(self._device).float()
        batch_size = states.shape[0]
        # indices to correlate Q network results back to batch record.
        batch_indices = np.arange(batch_size)

        # feed state to Q network to get Q values for that state
        q_values = self._q_net(states)[batch_indices, actions]

        # Q-learning
        # Instead of using dp table to get next q-values
        next_q_values = self.target_q_net(next_states).max(1)[0].detach() * (
                1 - terminals.squeeze())
        # print(next_q_values.shape)

        # q_values -> state, action values
        expected_q_values = next_q_values * self._discount + rewards.squeeze()

        loss = F.smooth_l1_loss(q_values, expected_q_values.float())
        self.log("training_loss", loss)
        # self.log("training_loss", loss)
        self._optimizer.zero_grad()
        loss.backward()

        # Page 7: www.nature.com/doifinder/10.1038/nature14236
        # "
        # We also found it helpful to clip the error term from the update γ*max a Q(S_{t+1}, a, θ_) - Q(S_t, A_t, θ)
        # to be between -1 and 1. Because the absolute value loss function jxj has a derivative of 21 for all negative
        # values of x and a derivative of 1 for all positive values of x, clipping the squared error to be between -1
        # and 1 corresponds to using an absolute value loss function for errors outside of the (-1,1) interval.
        # This form of error clipping further improved the stability of the algorithm.
        # "
        for param in self._q_net.parameters():
            # clamping as stated in Page-2: www.nature.com/doifinder/10.1038/nature14236
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()

    def policy(self, state: np.array):
        """
        :param: state
        :return the action
        """
        if not torch.is_tensor(state):
            state = torch.tensor(state).to(self._device)
        action_values = self._q_net(state).cpu().detach().unsqueeze(0).numpy()
        probs_batch = self.softmax(action_values, self._tau)

        # TODO: Discarding gumbel_softmax. If given more time would like to make this work.
        # action_values = self._q_net(state).cpu().detach().unsqueeze(0)
        # probs_batch = F.gumbel_softmax(action_values, self._tau).numpy()

        action = self._rand_generator.choice(self._num_actions, p=probs_batch.squeeze())
        return action

    def agent_start(self, state: np.array):
        """
        The first method called when the experiment starts, called after
        the environment starts.
        :param state: the state observation from the environment's env_start function.
        :return: The first action the agent takes.
        """
        self._sum_rewards = 0
        self._episode_steps = 0
        self._last_state = np.array(state)
        self._last_action = self.policy(self._last_state)
        return self._last_action

    def agent_step(self, reward: float, state: np.array):
        """
        A step taken by the agent.
        :param reward: the reward received for taking the last action taken
        :param state: the state observation from the environment's step based, where the agent ended up after the
                last step

        :return: The action the agent is taking.
        """
        self._sum_rewards += reward
        self._episode_steps += 1

        # Make state an array of shape (1, state_dim) to add a batch dimension and
        # to later match the get_action_values() and get_TD_update() functions
        state = np.array(state)

        # Select action
        action = self.policy(state)

        # Add new experience to replay buffer
        self._replay_buffer.append(self._last_state, self._last_action, reward, 0, state)

        # Experience replay:
        # Page-1: www.nature.com/doifinder/10.1038/nature14236
        # "During learning, we apply Q-learning updates, on samples (or mini-batches) of experience
        # (s,a,r,s9) , U(D), drawn uniformly at random from the pool of stored samples."
        if self._replay_buffer.size() > self._replay_buffer.mini_batch_size:
            # refresh the target network with snapshot of current network.
            # Page-2: www.nature.com/doifinder/10.1038/nature14236
            # "Second, we used an iterative update that adjusts the action-values (Q) towards target values that
            # are only periodically updated, thereby reducing correlations with the target."
            self.target_q_net.load_state_dict(self._q_net.state_dict())

            for _ in range(self._num_replay):
                # Get random sample experiences from the replay buffer
                experiences = self._replay_buffer.sample()
                self.training_step(experiences, self._episode_steps)

        # Update the last state and last action.
        # this prevents from this implementation from being parallelized, although will not worry about it right now.
        self._last_state = state
        self._last_action = action

        return action

    def agent_end(self, reward: float):
        """
        Run when the agent terminates.
        :param reward: the reward the agent received for entering the terminal state.
        """
        self._sum_rewards += reward
        self._episode_steps += 1

        # Set terminal state to an array of zeros
        state = np.zeros_like(self._last_state)

        # Append new experience to replay buffer
        self._replay_buffer.append(self._last_state, self._last_action, reward, 1, state)

        # Perform replay steps:
        if self._replay_buffer.size() > self._replay_buffer.mini_batch_size:
            self.target_q_net.load_state_dict(self._q_net.state_dict())
            for _ in range(self._num_replay):
                # Get sample experiences from the replay buffer
                experiences = self._replay_buffer.sample()
                self.training_step(experiences, 100000)

    def softmax(self, action_values: np.array, tau: float = 1.0):
        """
        What? Distilling the Knowledge in a Neural Network: https://arxiv.org/pdf/1503.02531.pdf
        Why? https://stackoverflow.com/questions/58764619/why-should-we-use-temperature-in-softmax
        Implementation:
        We use softmax(x) = softmax(x-c) identity to resolve possible overflow from exponential of large numbers in
        softmax
        :param action_values: A 2D array of shape (batch_size, num_actions).
                The action-values computed by an action-value network.
        :param tau: The temperature parameter scalar.
               𝜏 is the temperature parameter which controls how much the agent focuses on the highest valued
               actions. The smaller the temperature, the more the agent selects the greedy action.
               Conversely, when the temperature is high, the agent selects among actions more uniformly random.
        :return
            A 2D array of shape (batch_size, num_actions). Where each column is a probability distribution over
            the actions representing the policy.
        """
        # Compute the preferences by dividing the action-values by the temperature parameter tau
        preferences = action_values / tau
        # Compute the maximum preference across the actions
        max_preference = np.array([max(preference) / tau for preference in action_values])

        # Reshape max_preference array which has shape [Batch,] to [Batch, 1]. This allows NumPy broadcasting
        # when subtracting the maximum preference from the preference of each action.
        reshaped_max_preference = max_preference.reshape((-1, 1))

        # Compute the numerator, i.e., the exponential of the preference - the max preference.
        exp_preferences = np.array([np.exp(preference - max_preference) for preference, max_preference in
                                    zip(preferences, reshaped_max_preference)])
        # Compute the denominator, i.e., the sum over the numerator along the actions axis.
        sum_of_exp_preferences = exp_preferences.sum(axis=1)

        # Reshape sum_of_exp_preferences array which has shape [Batch,] to [Batch, 1] to  allow for NumPy broadcasting
        # when dividing the numerator by the denominator.
        reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))

        # Compute the action probabilities according to the equation in the previous cell.
        action_probs = exp_preferences / reshaped_sum_of_exp_preferences

        # squeeze() removes any singleton dimensions. It is used here because this function is used in the
        # agent policy when selecting an action (for which the batch dimension is 1.) As np.random.choice is used in
        # the agent policy and it expects 1D arrays, we need to remove this singleton batch dimension.
        action_probs = action_probs.squeeze()

        return action_probs

    def set_train(self):
        """
        Set networks into train mode
        """
        self._q_net.train()
        self.target_q_net.train()

    def set_eval(self):
        self._q_net.eval()
        self.target_q_net.eval()

    def save_checkpoint(self, episode_num, done=False):
        if done:
            checkpoint_name = os.path.join(self.checkpoint_dir, "final.pth")
        else:
            checkpoint_name = os.path.join(self.checkpoint_dir, f"ep_{episode_num}_step_{self._episode_steps}.pth")

        print('saving checkpoint...')
        checkpoint = {
            'network': self._q_net.state_dict(),
            'network_target': self.target_q_net.state_dict(),
            'optimizer': self._optimizer.state_dict()
        }
        torch.save(checkpoint, checkpoint_name)
        print(f"checkpoint saved at {checkpoint_name}")

    def get_latest_path(self):
        """
        Get the latest created file in the checkpoint directory
        :return the latest saved model weights
        """
        files = [fname for fname in os.listdir(self.checkpoint_dir) if fname.endswith(".pth")]
        filepaths = [os.path.join(self.checkpoint_dir, filepath) for filepath in files]
        latest_file = max(filepaths, key=os.path.getctime)
        return latest_file

    def load_checkpoint(self, checkpoint_path=None):
        """
        Load networks and optimizer parameters from checkpoint_path
        if checkpoint_path is None, use the latest created path from checkpoint_dir
        :param checkpoint_path: path to checkpoint
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_path()

        if os.path.isfile(checkpoint_path):
            key = 'cuda' if torch.cuda.is_available() else 'cpu'
            checkpoint = torch.load(checkpoint_path, map_location=key)
            self._q_net.load_state_dict(checkpoint['network'])
            self.target_q_net.load_state_dict(checkpoint['network_target'])
            self._optimizer.load_state_dict(checkpoint['optimizer'])

            print('checkpoint loaded at {}'.format(checkpoint_path))
        else:
            raise OSError("Checkpoint file not found.")

    @property
    def name(self):
        return self._name

    @property
    def sum_rewards(self):
        return self._sum_rewards
