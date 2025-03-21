import os
import esm.inverse_folding
from Bio import PDB
import pandas as pd
import numpy as np

def load_model():
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    return model.eval(), alphabet

def count_residues(fpath):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", fpath)
    chain_residue_counts = {}
    
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            residue_count = sum(1 for _ in chain.get_residues())
            chain_residue_counts[chain_id] = residue_count
    
    return chain_residue_counts

def find_min_chain(fpath):
    chain_residue_counts = count_residues(fpath)
    if not chain_residue_counts:
        raise ValueError(f"No chains found in {fpath}")
    return min(chain_residue_counts, key=chain_residue_counts.get)

def load_single_chain_structure(fpath, chain_id):
    structure = esm.inverse_folding.util.load_structure(fpath, chain_id)
    coords, seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
    if coords is None or seq is None:
        raise ValueError("Extracted coordinates or sequence is empty. Check your input PDB file.")
    return coords, seq

def score_sequence(model, alphabet, coords, seq):
    ll_fullseq, ll_withcoord = esm.inverse_folding.util.score_sequence(model, alphabet, coords, seq)
    perplexity = np.exp(-ll_fullseq)  # Compute perplexity from log-likelihood
    return ll_fullseq, ll_withcoord, perplexity

def process_pdb_files(directory):
    model, alphabet = load_model()
    log_data = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".pdb"):
            fpath = os.path.join(directory, filename)
            try:
                chain_id = find_min_chain(fpath)
                coords, seq = load_single_chain_structure(fpath, chain_id)
                ll_fullseq, ll_withcoord, perplexity = score_sequence(model, alphabet, coords, seq)
                log_data.append([filename, chain_id, ll_fullseq, ll_withcoord, perplexity])
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    log_df = pd.DataFrame(log_data, columns=["PDB File", "Chain", "Full Seq Log-Likelihood", "Log-Likelihood with Coordinates", "Perplexity"])
    print(log_df)
    return log_df

if __name__ == "__main__":
    pdb_directory = "/home/sdv/m1bi-ipfb/zfathi/Documents/stage/article/data/beta_amyloid_dimer_cf_1.5.5"
    log_df = process_pdb_files(pdb_directory)
    log_df.to_csv("pdb_log_results.csv", index=False)

