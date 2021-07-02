class Environment:
    def __init__(self, env):
        """
        Setup for the environment called when the environment first starts
        """
        super().__init__()
        self.env = env
        self.env.seed(0)
        self.reward_state_is_done = (0.0, None, False)

    def reset(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        :return: The first state state from the environment.
        """
        state = self.env.reset()
        self.reward_state_is_done = (0.0, state, False)

        # returns first state state from the environment
        return state

    def env_step(self, action):
        """
        A step taken by the environment.

        :param action:
            action: The action taken by the agent

        :return
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """
        current_state, reward, is_terminal, _ = self.env.step(action)

        self.reward_state_is_done = (reward, current_state, is_terminal)
        return self.reward_state_is_done
