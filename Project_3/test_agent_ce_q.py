from unittest import TestCase

import numpy as np

from agent_ce_q import CE_Q
from utils import error_plot, load_obj


class Test(TestCase):
    def test_ce_q(self):
        # CE_Q
        ce_q_errors, Q_1_ce, Q_2_ce, V_1_ce, V_2_ce, Pi_ce = CE_Q()

        error_plot(np.array(ce_q_errors), 'CE-Q')
        assert True

    def test_error_plot(self):
        d = load_obj('CE_Q')
        ce_q_errors = d['error_list']
        Q_1_ce = d['Q_1']
        Q_2_ce = d['Q_2']
        V_1_ce = d['V_1']
        V_2_ce = d['V_2']
        Pi_ce = d['Pi']

        error_plot(np.array(ce_q_errors), 'CE-Q')

        assert True
