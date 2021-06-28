from __future__ import absolute_import

import os

from .otb import ExperimentOTB
from ..datasets import TColor128


class ExperimentTColor128(ExperimentOTB):
    r"""Experiment pipeline and evaluation got10k_toolkit for TColor128 dataset.
    
    Args:
        root_dir (string): Root directory of TColor128 dataset.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """
    def __init__(self, root_dir,
                 result_dir='results', report_dir='reports'):
        self.dataset = TColor128(root_dir)
        self.result_dir = os.path.join(result_dir, 'TColor128')
        self.report_dir = os.path.join(report_dir, 'TColor128')
        # as nbins_iou increases, the success score
        # converges to the average overlap (AO)
        self.nbins_iou = 21
        self.nbins_ce = 51
