from unittest import TestCase

import numpy as np

from agent_foe_q import Foe_Q
from utils import error_plot, load_obj


class Test(TestCase):
    def test_foe_q(self):
        # Foe_Q
        foe_q_errors, Q_1, Q_2, V_1, V_2, Pi_1, Pi_2 = Foe_Q()

        error_plot(np.array(foe_q_errors), 'Foe-Q')
        assert True

    def test_error_plot(self):
        d = load_obj('Foe_Q_Working')
        q_errors = d['errors_list']
        Q_1 = d['Q_1']
        Q_2 = d['Q_2']
        V_1 = d['V_1']
        V_2 = d['V_2']
        Pi_1 = d['Pi_1']
        Pi_2 = d['Pi_2']

        error_plot(np.array(q_errors), 'Foe-Q')

        assert True
