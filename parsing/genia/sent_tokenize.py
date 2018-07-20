import subprocess

from shlex import quote

GENIA_SS_PATH = '~/bionlp_st_2013_supporting/src/geniass_ss.sh'


def genia_sent_tokenize(text):
    output = subprocess.check_output('source ~/.bash_profile;echo {} | {}'.format(quote(text), GENIA_SS_PATH),
                                     executable="/bin/bash",
                                     shell=True)
    output = output.decode('utf-8')
    output = output.splitlines()
    return output


if __name__ == '__main__':
    print(genia_sent_tokenize(
        "Induction of apoptosis of Beta cells of the pancreas by advanced glycation end-products , important mediators "
        "of chronic complications of diabetes mellitus . We herein report cytotoxicity of advanced glycation "
        "end-products ( AGEs ) on pancreatic beta cells . AGEs stimulated reactive oxygen species ( ROS ) generation "
        "but did not arrest proliferation of the P01308 -1 cell line . Pancreatic beta cell lines or primary cultured "
        "islets possess a receptor for P51606 ( RAGE ) , and its expression increased after P51606 treatment . "
        "TUNEL staining and FACS analysis using annexin V/PI antibodies showed that apoptosis increased in P01308 -1 "
        "cells or primary cultured islets when incubated with BSA conjugated with glyceraldehyde ( AGE2 ) "
        "or glucoaldehyde ( AGE3 ) , compared with those conjugated with glucose ( AGE1 ) . "
        "Reaction of P01308 -1 cells to Ki67 , which is a cellular marker for proliferation , "
        "was also increased after P51606 treatment . The ability of primary cultured islets to secrete insulin was "
        "retained even after P51606 treatment under either low or high glucose conditions . The antiserum against RAGE "
        "partially prevented P51606 -induced cellular events . Treatment of beta cells with the antioxidant "
        "metallothionein results in a significant reduction in pathologic changes . AGEs might be able to induce "
        "apoptosis as well as proliferation of pancreatic beta cell lines or primary cultured islets . Moreover , "
        "antibody array showed that Q06609 and P43351 were significantly decreased in AGE2-treated P01308 -1 cells . "
        "AGEs might inhibit homologous DNA recombination for repairing DNA of P01308 -1 cells damaged by ROS "
        "generation .It might be suggested that treatment of AGEs resulted in ROS production and apoptosis through "
        "their receptor on pancreatic beta cells . AGEs might deteriorate function of pancreatic beta cells in "
        "patients with long-term hyperglycemia ."))
