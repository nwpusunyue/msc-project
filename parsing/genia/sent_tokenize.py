import subprocess

from shlex import quote

GENIA_SS_PATH = '~/bionlp_st_2013_supporting/src/geniass_ss.sh'


def genia_sent_tokenize(text):
    output = subprocess.check_output(
        'source ~/.bash_profile;echo {} | {}'.format(quote(text), GENIA_SS_PATH).encode('utf-8'),
        executable="/bin/bash",
        shell=True)
    output = output.decode('utf-8').strip()
    if output[:2] == "b'" or output[:2] == 'b"':
        output = output[2:-1]
    output = output.splitlines()
    return output


if __name__ == '__main__':
    print(genia_sent_tokenize(
        "AB-CHMINACA , AB-PINACA , and FUBIMINA : Affinity and Potency of Novel Synthetic Cannabinoids in Producing Δ9- "
        "ent_x -Like Effects in Mice . Diversion of synthetic cannabinoids for abuse began in the early 2000s . "
        "Despite legislation banning compounds currently on the drug market , illicit manufacturers continue to "
        "release new compounds for recreational use . This study examined new synthetic cannabinoids , AB-CHMINACA "
        "( N-[1-amino-3-methyl-oxobutan-2-yl]-1-[cyclohexylmethyl]-1H-indazole-3-carboxamide ) , AB-PINACA "
        "[ N-(1-amino-3-methyl-1-oxobutan-2-yl)-1-pentyl-1H-indazole-3-carboxamide ] , and FUBIMINA "
        "[ ( 1-(5-fluoropentyl)-1H-benzo[d]imadazol-2-yl ) (naphthalen-1-yl)methanone ] , with the hypothesis that "
        "these compounds , like those before them , would be highly susceptible to abuse . "
        "Cannabinoids were examined in vitro for binding and activation of ent_x receptors , and in vivo for "
        "pharmacological effects in mice and in Δ(9)-tetrahydrocannabinol ( Δ(9)-THC ) discrimination . "
        "AB-CHMINACA , AB-PINACA , and FUBIMINA bound to and activated ent_x and CB2 receptors , and produced "
        "locomotor suppression , antinociception , hypothermia , and catalepsy . Furthermore , these compounds , "
        "along with JWH-018 [ 1-pentyl-3-(1-naphthoyl)indole ] , ent_x , 497 [ rel-5-(1,1-dimethylheptyl)-2- [ ( 1R,3S ) "
        "-3-hydroxycyclohexyl ] -phenol ] , and WIN55,212-2 ( [ ( 3R ) "
        "-2,3-dihydro-5-methyl-3-(4-morpholinylmethyl)pyrrolo[1,2,3-de]-1,4-benzoxazin-6-yl ] -1-naphthalenyl-methanone "
        ", monomethanesulfonate ) , substituted for Δ(9)-THC in Δ(9)-THC discrimination . Rank order of potency "
        "correlated with ent_x receptor-binding affinity , and all three compounds were full agonists in "
        "[(35)S]GTPγS binding , as compared with the partial agonist Δ(9)-THC . Indeed , AB-CHMINACA and AB-PINACA "
        "exhibited higher efficacy than most known full agonists of the ent_x receptor . Preliminary analysis of "
        "urinary metabolites of the compounds revealed the expected hydroxylation . AB-PINACA and AB-CHMINACA are "
        "of potential interest as research tools due to their unique chemical structures and high ent_x receptor "
        "efficacies . Further studies on these chemicals are likely to include research on understanding cannabinoid "
        "receptors and other components of the endocannabinoid system that underlie the abuse of synthetic cannabinoids ."))
