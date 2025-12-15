import os
import re
import argparse
from multiprocessing import Pool


def run_tmalign(query_pdb, target_pdb, output_file, tmalign_path):
    cmd = f"{tmalign_path} {query_pdb} {target_pdb} > {output_file}"
    os.system(cmd)


def extract_rmsd_from_output(output_file):
    if not os.path.exists(output_file):
        return "N/A"

    with open(output_file, 'r') as f:
        for line in f:
            if "RMSD=" in line:
                parts = line.split(',')
                for part in parts:
                    if "RMSD=" in part:
                        return part.split('=')[1].strip()
    return "N/A"


def parse_hmmer_file(file_path):
    protein_info = {}
    current_round = 1
    first_round_proteins = []
    round_identified = False

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("@@ Round:"):
                match = re.search(r"Round:\s+(\d+)", line)
                if match:
                    current_round = int(match.group(1))
                    round_identified = True

            match = re.search(r"afdb50_([^\s]+)", line)
            if match:
                protein_id = match.group(1)
                if protein_id not in protein_info:
                    protein_info[protein_id] = current_round
                    if not round_identified or current_round == 1:
                        first_round_proteins.append(protein_id)

    all_proteins = list(protein_info.keys())
    return first_round_proteins, all_proteins


def parse_m8_file(file_path):
    protein_ids = []
    seen = set()
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r"afdb50_([^\s]+)", line)
            if match:
                pid = match.group(1)
                if pid not in seen:
                    protein_ids.append(pid)
                    seen.add(pid)

    return protein_ids, protein_ids


def get_pdb_path(base_dir, protein_id):
    if len(protein_id) < 6:
        return None
    dir1 = protein_id[0:2]
    dir2 = protein_id[2:4]
    dir3 = protein_id[4:6]
    return os.path.join(base_dir, dir1, dir2, dir3, f"{protein_id}.pdb")


def process_alignment(args):
    threads_per_task = args.threads_per_task
    input_file = args.input_file
    output_dir = args.output_dir
    afdb_pdb_dir = args.afdb_pdb_dir
    tmalign_path = args.tmalign_path
    mode = args.mode

    os.makedirs(output_dir, exist_ok=True)

    if mode == 'hmmer':
        candidate_refs, all_proteins = parse_hmmer_file(input_file)
    elif mode == 'm8':
        candidate_refs, all_proteins = parse_m8_file(input_file)
    else:
        print(f"Unknown mode: {mode}")
        return

    if len(all_proteins) < 2:
        print(f"Insufficient proteins found in {input_file} (at least 2 required, current: {len(all_proteins)})")
        return

    ref_protein = None
    ref_pdb_path = None

    for pid in candidate_refs:
        pdb_path = get_pdb_path(afdb_pdb_dir, pid)
        if pdb_path and os.path.exists(pdb_path):
            ref_protein = pid
            ref_pdb_path = pdb_path
            break

    if not ref_protein:
        print(f"No valid reference protein PDB file found (among {len(candidate_refs)} candidates)")
        return

    print(f"[{mode}] Using reference protein: {ref_protein}")
    print(f"Total proteins to align: {len(all_proteins)}")

    tasks = []
    processed_targets = set()
    processed_targets.add(ref_protein)

    for pid in all_proteins:
        if pid in processed_targets:
            continue

        target_pdb = get_pdb_path(afdb_pdb_dir, pid)
        if target_pdb and os.path.exists(target_pdb):
            output_file = os.path.join(output_dir, f"{pid}.rmsd")
            tasks.append((ref_pdb_path, target_pdb, output_file, tmalign_path))
            processed_targets.add(pid)
        else:
            pass

    if tasks:
        print(f"Starting alignment of {len(tasks)} structures (Threads: {threads_per_task})...")
        with Pool(threads_per_task) as p:
            p.starmap(run_tmalign, tasks)

        success_count = 0
        for _, _, out_f, _ in tasks:
            if os.path.exists(out_f) and os.path.getsize(out_f) > 0:
                success_count += 1
        print(f"Alignment complete: {success_count}/{len(tasks)} RMSD files successfully generated.")

    else:
        print("No valid alignment tasks generated (target PDBs might be missing).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process m8 or hmmer files and calculate RMSD')
    parser.add_argument('--mode', type=str, choices=['m8', 'hmmer'], required=True, help='Input file type: m8 or hmmer')
    parser.add_argument('--input_file', type=str, required=True, help='Input MSA file path (m8 or hmmer)')
    parser.add_argument('--output_dir', type=str, required=True, help='RMSD output directory')
    parser.add_argument('--afdb_pdb_dir', type=str, required=True, help='AFDB50_pdb database directory')
    parser.add_argument('--tmalign_path', type=str, required=True, help='TMalign executable path')
    parser.add_argument('--threads_per_task', type=int, default=10, help='Number of parallel threads')

    args = parser.parse_args()
    process_alignment(args)