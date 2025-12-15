import numpy as np
import os
import sys

path_tran = sys.argv[1]
pdb_index = sys.argv[2]

TMscore = 0

input_file_path = os.path.join(path_tran+'/tran_files/tmscore_tran')
input_file_path2 = os.path.join(path_tran+'/tran_files/tmscore_qianhou')
output_file_path = os.path.join(path_tran+'/tran_files/tmscore_rot_metrc')
try:
    if not os.path.exists(input_file_path):
        raise Exception(f"File {input_file_path} dose not exist")

    with open(input_file_path, "r") as f:
        lines = f.readlines()
        matrix_lines = lines[-11:-8]

    with open(input_file_path2, "r") as f:
        lines = f.readlines()
        TMscore = lines[-18:-17]

    numbers = []
    for line in matrix_lines:
        line = line.strip()
        if line:
            numbers += line.split()

    with open(output_file_path, "w") as f:
        f.write(f"{numbers[2]}  {numbers[7]}  {numbers[12]}  ")
        f.write(f"{numbers[3]}  {numbers[8]}  {numbers[13]}  ")
        f.write(f"{numbers[4]}  {numbers[9]}  {numbers[14]}  ")
        f.write(f"{numbers[1]}  {numbers[6]}  {numbers[11]}  ")
except Exception as e:
    print(f"An error occurred while processing file {input_file_path}：{e}")

rot_to_pdb = np.zeros((3, 3))
trans = np.zeros(3)

line = []
with open(output_file_path, 'r') as f:
    for line in f.readlines():
        line = np.array(line.strip().split('  '))

for i in range(3):
    for j in range(3):
        rot_to_pdb[i][j] = float(line[3 * i + j])

    trans[i] = float(line[9 + i])

with open(path_tran+'/stage2/Step_'+pdb_index+'_sample.pdb', 'r') as f, open(path_tran+'/tran_out_pdb/model_'+pdb_index+'.pdb', 'w') as outfile:
    for line in f.readlines():
        if (line[0:6] != "ATOM  "):
            continue

        str1 = line[0:31]
        str2 = line[54:80]

        str_x = line[30:38]
        str_y = line[38:46]
        str_z = line[46:54]

        atom_x = float(line[30:38])
        atom_y = float(line[38:46])
        atom_z = float(line[46:54])
        coordinate = np.array([atom_x, atom_y, atom_z])
        coordinate_new = rot_to_pdb.dot(coordinate- trans)

        str_x = str(round(coordinate_new[0], 2))
        str_y = str(round(coordinate_new[1], 2))
        str_z = str(round(coordinate_new[2], 2))

        for i in range(8 - len(str_x)):
            str_x += ' '

        for i in range(8 - len(str_y)):
            str_y += ' '

        for i in range(8 - len(str_z)):
            str_z += ' '

        new_line = str1 + str_x + str_y + str_z + str2 + '\n'
        outfile.write(new_line)
    outfile.write('TER')
