import os
import subprocess

from nltk.tokenize.api import TokenizerI
from shlex import quote, split

GTB_PATH = '~/bionlp_st_2013_supporting/tls/GTB-tokenize.pl'


class GTBTokenizer(TokenizerI):

    def __init__(self):
        pass

    def tokenize(self, text):
        output = subprocess.check_output('echo {} | {}'.format(quote(text), GTB_PATH).encode('utf-8'),
                                         executable="/bin/bash",
                                         shell=True)
        output = output.decode('utf-8').strip()
        if output[:2] == "b'" or output[:2] == 'b"':
            output = output[2:-1]
        output = output.split(' ')
        return output

    def span_tokenize(self, s):
        pass


if __name__ == '__main__':
    print(GTBTokenizer().tokenize("Let's β tokenize the hell out of this!"))
