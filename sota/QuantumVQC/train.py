import argparse
import importlib
import os
import sys

SOTA_DIR = os.path.dirname(os.path.abspath(__file__))


def get_args():
    parser = argparse.ArgumentParser()
    # Runline used by run_improved.py
    parser.add_argument('--model', type=str, default="network", help="module name of the variant")
    parser.add_argument('--variant_dir', type=str, default=SOTA_DIR, help="directory holding the variant")
    # ExquisiteNetV2-style flags
    parser.add_argument('-network', type=str, default=None, help="dotted module path, e.g. models.network_X")
    parser.add_argument('-data', type=str, default=None, help="path to seeds_dataset.txt")
    parser.add_argument('-bs', type=int, default=None, help="ignored")
    parser.add_argument('-amp', action="store_true", help="ignored")
    args, _ = parser.parse_known_args()
    return args


def gene_id_from(module_name):
    name = module_name.split('.')[-1]
    for prefix in ('network_', 'model_'):
        if name.startswith(prefix):
            return name[len(prefix):]
    return 'seed'


if __name__ == '__main__':
    args = get_args()

    if args.network:
        sys.path.insert(0, SOTA_DIR)
        module_name = args.network
    else:
        sys.path.insert(0, os.path.abspath(args.variant_dir))
        module_name = args.model
    module = importlib.import_module(module_name)
    gene_id = gene_id_from(module_name)

    data_path = args.data or os.path.join(SOTA_DIR, 'data', 'seeds_dataset.txt')
    results_dir = os.path.join(SOTA_DIR, 'results')
    metrics = module.main(gene_id=gene_id, data_path=data_path, results_dir=results_dir)

    print(f"obj1={metrics['obj1']:.4f} obj2={metrics['obj2']}")
    print(f"results have been written to {results_dir}/{gene_id}_results.csv")
    print('='*120);print('job done');print('='*120)
