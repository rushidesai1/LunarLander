from unittest import TestCase

import numpy as np

from agent_q_learning import Q_learning
from utils import error_plot, load_obj


class Test(TestCase):
    def test_q_learning(self):
        # Q-learning
        q_learning_errors, Q_1_q_learning, Q_2_q_learning = Q_learning()

        error_plot(np.array(q_learning_errors), 'Q_Learning')
        assert True

    def test_error_plot(self):
        d = load_obj('Q_Learning_Working')
        q_errors = d['error_list']
        Q_1 = d['Q_1']
        Q_2 = d['Q_2']

        error_plot(np.array(q_errors), 'Q_Learning')

        assert True
