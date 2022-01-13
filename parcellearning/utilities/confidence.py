import argparse
import numpy as np
from niio import loaded, write

def main(args):

    print(f'Prob file: {args.probabilities}')

    P = loaded.load(args.probabilities)
    P = np.nanmax(P, axis=1)

    write.save(P, args.output, 'L')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compute the model confidence i.e. maximum probability estimate, for each vertex.')

    parser.add_argument('-p', 
                        '--probabilities',
                        help='Estimated model probabilities.',
                        required=True,
                        type=str)
    
    parser.add_argument('-o',
                        '--output',
                        required=True,
                        type=str)
    
    args = parser.parse_args()
    main(args)
    