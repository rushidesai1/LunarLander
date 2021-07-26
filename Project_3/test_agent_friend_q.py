from unittest import TestCase

import numpy as np

from agent_friend_q import Friend_Q
from utils import error_plot, load_obj


class Test(TestCase):
    def test_friend_q(self):
        # Friend-Q
        friend_q_errors, Q_1_friend, Q_2_friend = Friend_Q()

        error_plot(np.array(friend_q_errors), 'Friend_Q_Learning')
        # based on Taka's comment in office hours, just dropping the zeros from plot.
        # Frankly I don't quite understand why are we dropping zeros. Maybe thats what paper did and to replicate it we
        # need to drop as well
        error_plot(np.array(friend_q_errors)[np.where(np.array(friend_q_errors) > 0)], 'Friend_Q_Learning')
        assert True

    def test_error_plot(self):
        d = load_obj('Friend_Q_Learning_Working')
        q_errors = d['error_list']
        Q_1 = d['Q_1']
        Q_2 = d['Q_2']

        error_plot(np.array(q_errors), 'Friend_Q_Learning')
        # if you remove zeros
        error_plot(np.array(q_errors)[np.where(np.array(q_errors) > 0)], 'Friend_Q_Learning')

        assert True
