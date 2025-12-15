import sys

path_pdb = sys.argv[1]
index = sys.argv[2]
path_out = sys.argv[3]

with open(path_pdb+'/model_'+str(int(index))+'.pdb')as f, open(path_out+'/model_movie_sample0.pdb', "a+") as outfile:
    outfile.write('MODEL        '+ str(int(index)) + '\n')
    for line in f:
        ATOM = line.split()[0]
        if ATOM=="ATOM":
            outfile.write(line)
        if ATOM=="END" or ATOM=="TER":
            outfile.write('ENDMDL' + '\n')
            break

