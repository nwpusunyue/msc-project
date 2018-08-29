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
    print(GTBTokenizer().tokenize("Furthermore , these compounds , along with JWH-018 [ 1-pentyl-3-(1-naphthoyl)indole "
                                  "] , ent_x , 497 [ rel-5-(1,1-dimethylheptyl)-2- [ ( 1R,3S ) -3-hydroxycyclohexyl "
                                  "] -phenol ] , and WIN55,212-2 ( [ ( 3R ) "
                                  "-2,3-dihydro-5-methyl-3-(4-morpholinylmethyl)pyrrolo[1,2,3-de]-1,4-benzoxazin-6-yl ] "
                                  "-1-naphthalenyl-methanone , monomethanesulfonate ) , "
                                  "substituted for (9)-THC in (9)-THC discrimination .!"))
