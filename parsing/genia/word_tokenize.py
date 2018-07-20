import subprocess

from nltk.tokenize.api import TokenizerI
from shlex import quote

GTB_PATH = '~/bionlp_st_2013_supporting/tls/GTB-tokenize.pl'


class GTBTokenizer(TokenizerI):

    def __init__(self):
        pass

    def tokenize(self, text):
        output = subprocess.check_output('echo {} | {}'.format(quote(text), GTB_PATH),
                                         executable="/bin/bash",
                                         shell=True)
        output = output.decode('utf-8').strip()
        output = output.split(' ')
        return output

    def span_tokenize(self, s):
        pass


if __name__ == '__main__':
    print(GTBTokenizer().tokenize("Let's tokenize the hell out of this!"))
