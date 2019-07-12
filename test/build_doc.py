import os, sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent))

from bert.preprocess.preprocess import build_dictionary

train_path = '../data/example'
dictionary_path = 'build_doc.TEST'

def main():
    print('[TEST] build document')
    print('format:')
    print('A T C G . . .')
    print('A G C T . . .')
    build_dictionary(train_path, dictionary_path)

if __name__ == '__main__':
   main()
