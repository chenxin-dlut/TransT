from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import sys

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from pysot_toolkit.toolkit.datasets import OTBDataset, UAVDataset, LaSOTDataset, VOTDataset, NFSDataset, VOTLTDataset
from pysot_toolkit.toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, EAOBenchmark, F1Benchmark
from pysot_toolkit.toolkit.visualization import draw_success_precision
import numpy as np
parser = argparse.ArgumentParser(description='transt evaluation')
parser.add_argument('--tracker_path', '-p', type=str, default='',
                    help='tracker result path')
parser.add_argument('--dataset', '-d', type=str, default='LaSOT',
                    help='dataset name')
parser.add_argument('--num', '-n', default=1, type=int,
                    help='number of thread to eval')
parser.add_argument('--tracker_prefix', '-t', default='',
                    type=str, help='tracker name')
parser.add_argument('--show_video_level', '-s', dest='show_video_level',
                    action='store_true')
parser.add_argument('--vis', dest='vis', action='store_true')
parser.set_defaults(show_video_level=False)
args = parser.parse_args()

def main():

    tracker_dir = os.path.join(args.tracker_path, args.dataset)
    trackers = glob(os.path.join(args.tracker_path,
                                 args.dataset,
                                 args.tracker_prefix+'*'))
    trackers = [x.split('/')[-1] for x in trackers]

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    root = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                         'testing_dataset'))
    root = os.path.join(root, args.dataset)
    if 'OTB' in args.dataset:
        dataset = OTBDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                if attr == 'ALL':
                    draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)
    elif 'LaSOT' == args.dataset:
        dataset = LaSOTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        norm_precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision,
                trackers), desc='eval norm precision', total=len(trackers), ncols=100):
                norm_precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
                show_video_level=args.show_video_level)
        if args.vis:
            draw_success_precision(success_ret,
                                   name=dataset.name,
                                   videos=dataset.attr['ALL'],
                                   attr='ALL',
                                   precision_ret=precision_ret,
                                   norm_precision_ret=norm_precision_ret)
    elif 'UAV' in args.dataset:
        dataset = UAVDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                if attr == 'ALL':
                    draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)
    elif 'NFS' in args.dataset:
        dataset = NFSDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                if attr == 'ALL':
                    draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)
    elif args.dataset in ['VOT2016', 'VOT2017', 'VOT2018', 'VOT2019']:
        dataset = VOTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        ar_benchmark = AccuracyRobustnessBenchmark(dataset)
        ar_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(ar_benchmark.eval,
                trackers), desc='eval ar', total=len(trackers), ncols=100):
                ar_result.update(ret)

        benchmark = EAOBenchmark(dataset)
        eao_result = {}
        EAO_list = [] # newly added (2020.07.05)
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                trackers), desc='eval eao', total=len(trackers), ncols=100):
                eao_result.update(ret)
        for name in eao_result:
            EAO_list.append(eao_result[name]['all'])
        mean_eao = np.mean(np.array(EAO_list))
        print('Mean EAO = ',mean_eao)
        ar_benchmark.show_result(ar_result, eao_result,
                show_video_level=args.show_video_level)
    elif 'VOT2018-LT' == args.dataset:
        dataset = VOTLTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = F1Benchmark(dataset)
        f1_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                trackers), desc='eval f1', total=len(trackers), ncols=100):
                f1_result.update(ret)
        benchmark.show_result(f1_result,
                show_video_level=args.show_video_level)


if __name__ == '__main__':
    main()