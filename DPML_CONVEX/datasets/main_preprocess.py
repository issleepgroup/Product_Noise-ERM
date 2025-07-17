import argparse

from preprocess import (
    adult, covertype, gisette, kddcup99, mnist, realsim, rcv1, syntheticH)


CACHE_LOCATION = 'data_cache'
OUTPUT_LOCATION = 'data'

datasets = {
    'adult': adult,
    'covertype': covertype,
    'gisette': gisette, #Sparse, but can be represented in a dense format
    'kddcup99': kddcup99,
    'mnist': mnist,
    'realsim': realsim,
    'rcv1': rcv1,
    'syntheticH':syntheticH,
}


def preprocess_all():
    for (name, dataset) in datasets.items():
        print('Preprocessing ' + name)
        dataset.preprocess(CACHE_LOCATION, OUTPUT_LOCATION)



def main():
    parser = argparse.ArgumentParser(
        description='Preprocess the specified dataset')
    parser.add_argument('dataset', type=str, help='The name of the dataset',
                        choices=list(datasets.keys()) + ["all"])
    args = parser.parse_args()

    if args.dataset == 'all':
        preprocess_all()
    elif args.dataset == 'syntheticH':
        datasets[args.dataset].preprocess(OUTPUT_LOCATION)
    else:
        datasets[args.dataset].preprocess(CACHE_LOCATION, OUTPUT_LOCATION)


if __name__ == '__main__':
    main()
