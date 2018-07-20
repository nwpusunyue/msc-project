import subprocess

from shlex import quote

GTB_PATH = '~/bionlp_st_2013_supporting/tls/GTB-tokenize.pl'


def genia_word_tokenize(text):
    output = subprocess.check_output('echo {} | {}'.format(quote(text), GTB_PATH),
                                     executable="/bin/bash",
                                     shell=True)
    output = output.decode('utf-8').strip()
    output = output.split(' ')
    return output


if __name__ == '__main__':
    print(genia_word_tokenize("Let's tokenize the hell out of this!"))
