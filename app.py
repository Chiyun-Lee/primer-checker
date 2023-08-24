import streamlit as st

import numpy as np
import pandas as pd
import malariagen_data

from typing import Iterable, Dict, Any, List, Tuple
from numpy.typing import NDArray
from xarray.core.dataset import Dataset

@st.cache_data
def cache_pf7_resources():
    pf7 = malariagen_data.Pf7("gs://pf7_release/")
    metadata = pf7.sample_metadata()
    callset = pf7.variant_calls()
    return pf7, metadata, callset

@st.cache_data
def cache_read_csv(f: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    return pd.read_csv(f)

def get_sequence_reverse_complement(
        s: Iterable,
        as_str: bool=False
):
    
    base_to_complement = {
        'A': 'T', 'C': 'G', 'T': 'A', 'G': 'C',
        'a': 't', 'c': 'g', 't': 'a', 'g': 'c'
    }
    rcs = list(reversed([base_to_complement[b] for b in s]))
    if as_str:
        return "".join(rcs)
    return rcs

def streamlit_find_sequence_in_reference_region(
        sequence,
        chromosome: str,
        start_locus: int,
        end_locus: int,
        top_k: int=-1,
        save_matches: bool=True,
        check_reverse: bool=True,
) -> pd.DataFrame:
    
    pf7, _, _ = cache_pf7_resources()
    
    if isinstance(sequence, str):
        sequence = np.array(list(sequence))
    sequence = np.char.lower(sequence)
    ref_sequence = pf7.genome_sequence(
        region=[f'{chromosome}:{start_locus}-{end_locus}']
    ).compute().astype(str)
    
    k = len(sequence)
    num_comparisons = len(ref_sequence) - k + 1
    
    starts = np.full(num_comparisons * (1 + check_reverse), -1)
    ends = np.full(num_comparisons * (1 + check_reverse), -1)
    diffs = np.full(num_comparisons * (1 + check_reverse), -1)
    seqs =  np.full(num_comparisons * (1 + check_reverse), -1, dtype=object)
    direction = np.full(num_comparisons * (1 + check_reverse), -1, dtype=object)
    
    for i, l in enumerate(range(num_comparisons)):
        num_diffs = np.sum(ref_sequence[l:(l+k)] != sequence)
        starts[i] = start_locus + l
        ends[i] = start_locus + l + k - 1
        diffs[i] = num_diffs
        direction[i] = 'forward'
        if save_matches:
            seqs[i] = "".join(ref_sequence[l:(l+k)])
    
    if check_reverse:
        reverse_complement_seq = np.array(
            get_sequence_reverse_complement(
                sequence,
                as_str=False))
        for i, l in enumerate(range(num_comparisons)):
            num_diffs = np.sum(ref_sequence[l:(l+k)] != reverse_complement_seq)
            starts[num_comparisons + i] = start_locus + l
            ends[num_comparisons + i] = start_locus + l + k - 1
            diffs[num_comparisons + i] = num_diffs
            direction[num_comparisons + i] = 'reverse'
            if save_matches:
                seqs[num_comparisons + i] = "".join(ref_sequence[l:(l+k)])
    
    hits_df = pd.DataFrame({
        'chromosome': chromosome,
        'start': starts,
        'end': ends,
        'differences': diffs,
        'direction': direction
    })
    if save_matches:
        hits_df['match'] = seqs
    
    hits_df.sort_values(by=['differences','start'],
                        inplace=True)
    if top_k != -1:
        return hits_df.head(top_k)
    return hits_df

def streamlit_find_primer_coordinates(
        primer_data: pd.DataFrame,
        search_margin_bp: int=150,
        amplicon_id_column: str='amplicon_id',
        forward_seq_column: str='forward_primer_seq',
        reverse_seq_column: str='reverse_primer_seq',
        gene_start_locus_column: str='gene_start_coordinate',
        gene_end_locus_column: str='gene_end_coordinate',
        chromosome_column: str='chromosome',
) -> pd.DataFrame:
    
    amplicon_primer_coords = []
    for row in primer_data.iterrows():

        r = row[1]
        forward_seq = np.array(list(r[forward_seq_column]))
        reverse_seq = np.array(list(r[reverse_seq_column]))
        
        top_hit_forward = streamlit_find_sequence_in_reference_region(
            forward_seq,
            r[chromosome_column],
            r[gene_start_locus_column] - search_margin_bp,
            r[gene_end_locus_column] + search_margin_bp,
            top_k=1
        )
        
        top_hit_forward.insert(0, 'primer_id', f'{r[amplicon_id_column]}-forward')
        if top_hit_forward['direction'].item() == 'reverse':
            top_hit_forward['sequence'] = get_sequence_reverse_complement(forward_seq, as_str=True)
        else:
            top_hit_forward['sequence'] = "".join(forward_seq)
        
        top_hit_reverse = streamlit_find_sequence_in_reference_region(
            reverse_seq,
            r[chromosome_column],
            r[gene_start_locus_column] - search_margin_bp,
            r[gene_end_locus_column] + search_margin_bp,
            top_k=1
        )
        
        top_hit_reverse.insert(0, 'primer_id', f'{r[amplicon_id_column]}-reverse')
        if top_hit_reverse['direction'].item() == 'reverse':
            top_hit_reverse['sequence'] = get_sequence_reverse_complement(reverse_seq, as_str=True)
        else:
            top_hit_reverse['sequence'] = "".join(reverse_seq)
        
        amplicon_primer_coords.append(pd.concat([top_hit_forward, top_hit_reverse]))
    
    return pd.concat(amplicon_primer_coords).reset_index()

def streamlit_parse_sequence_info(sequence_info) -> Dict[str, Dict[str, Any]]:
    if isinstance(sequence_info, str):
        sequence_info = pd.read_csv(sequence_info)
    seq_id_to_info = {}

    for rows in sequence_info.iterrows():
        si = rows[1]
        seq_info = {
            'id': si['primer_id'],
            'chromosome': si['chromosome'],
            'start_locus': si['start'],
            'end_locus': si['end'],
            'sequence': si['sequence']
        }
        seq_id_to_info[seq_info['id']] = seq_info

    return seq_id_to_info

def process_alleles(alleles: NDArray[str]) -> pd.DataFrame:
    called_alleles = alleles[alleles != '']
    lengths = np.array([len(a) for a in called_alleles])
    
    ref = called_alleles[0]
    ref_length = lengths[0]
    
    types = ['reference']
    for i in range(1, len(called_alleles)):
        if called_alleles[i] == '*':
            types.append('spanning-deletion')
            lengths[i] = 0
        elif lengths[i] < ref_length:
            types.append('deletion')
        elif lengths[i] > ref_length:
            types.append('insertion')
        else:
            diffs = (np.array(list(called_alleles[i])) != np.array(list(ref))).sum()
            if diffs == 1:
                types.append('SNP')
            else:
                types.append('MNP')

    return pd.DataFrame({
        'allele': called_alleles,
        'type': types,
        'length': lengths,
        'length_diff': lengths - ref_length
    })

def count_alleles(calls: NDArray[int]):
    is_hom = calls[:, 0] == calls[: , 1]
    ic = calls.copy()
    ic[is_hom, 1] = -2
    
    unique_alleles, counts = np.unique(ic, return_counts=True)
    
    counts_df = pd.DataFrame({
        'allele_code': unique_alleles,
        'counts': counts
    }).query('allele_code != -2')

    return counts_df.set_index(counts_df.allele_code)

def streamlit_assess_variation_in_region(
        chromosome: str,
        start_locus: int,
        end_locus: int,
        samples_ids: List[str]
) -> Dict[Tuple[str, int], pd.DataFrame]:
    
    _, _, callset = cache_pf7_resources()
    
    variant_mask = (
        (callset.variant_chrom == chromosome) &
        np.isin(callset.variant_position, list(range(start_locus, end_locus + 1))))
    variant_mask = variant_mask.compute().data
    
    loci_considered = callset.variant_position[variant_mask].compute().data
    is_locus_hq_pass = callset.variant_filter_pass[variant_mask].compute().data
    
    sample_to_index = {s: i for i, s in enumerate(callset.sample_id.data.compute())}
    callset_samples_indices = [sample_to_index[s] for s in samples_ids]
    
    # global_progressbar.update(f"Processing {len(loci_considered)} variants")
    
    calls = callset.call_genotype[variant_mask, callset_samples_indices, :].compute().data
    alleles = callset.variant_allele[variant_mask, :].compute().data
    locus_to_counts = {}

    for i, l in enumerate(loci_considered):
        variant_info_df = process_alleles(alleles[i, :])
        variant_info_df['qc_pass'] = is_locus_hq_pass[i]
        
        counts_df = count_alleles(calls[i, :, :])
        
        variant_info_df['counts'] = counts_df[counts_df.allele_code != -1]['counts']
        variant_info_df['counts'].fillna(0, inplace=True, downcast='int')
        total_counts = variant_info_df['counts'].sum()
        
        num_missing = 0
        if -1 in counts_df.allele_code:
            num_missing = counts_df[counts_df.allele_code == -1]['counts'].item()
        variant_info_df = pd.concat([
            variant_info_df,
            pd.DataFrame({
                'allele': '-',
                'type': 'missing',
                'length': 0,
                'length_diff': 0,
                'qc_pass': is_locus_hq_pass[i],
                'counts': num_missing
            }, index=[-1])], ignore_index=False)
        
        variant_info_df['freq'] = np.round(variant_info_df['counts']/total_counts, 5)
        
        variant_info_df.loc[-1, 'freq'] = np.round(variant_info_df.loc[-1, 'counts']/total_counts, 5)

        locus_to_counts[(chromosome, l)] = variant_info_df

    return locus_to_counts

def streamlit_assess_variation_in_sequence(
        sequence_info: Dict[str, Any],
        samples_ids: List[str],
        only_indels: bool=False,
        only_pass_variants: bool=False
) -> pd.DataFrame:
    
    if (sequence_info['start_locus'] + len(sequence_info['sequence']) - 1) != sequence_info['end_locus']:
        raise ValueError('Sequence length incoherent with start/end locus')
    
    locus_to_dfs = streamlit_assess_variation_in_region(
        chromosome=sequence_info['chromosome'],
        start_locus=sequence_info['start_locus'],
        end_locus=sequence_info['end_locus'],
        samples_ids=samples_ids
    )
    
    for l, summary_df in locus_to_dfs.items():
        summary_df.insert(0, 'locus', l[1])
        summary_df.insert(0, 'chromosome', l[0])
        summary_df.insert(0, 'sequence_id', sequence_info['id'])
    
    variation_summary_df = pd.concat(list(locus_to_dfs.values()), ignore_index=True)
    
    if len(variation_summary_df) > 0:
        loci_indices_in_seq = variation_summary_df.locus.to_numpy() - sequence_info['start_locus']
        variation_summary_df.insert(3, 'seq_allele', np.array(list(sequence_info['sequence']))[loci_indices_in_seq])

    return variation_summary_df

@st.cache_data
def streamlit_run_workflow_pf7(country, significant_variation_frequency):    
    primer_data = cache_read_csv(uploaded_csv)
    pf7, metadata, callset = cache_pf7_resources()
    
    primer_coords_df = streamlit_find_primer_coordinates(primer_data)
    primers_info = streamlit_parse_sequence_info(primer_coords_df)
    
    # global_progressbar.update("Gathering data")
    
    if country == "All countries":
        samples_ids = metadata.Sample.to_numpy()
    else:
        samples_ids = metadata.query(f'Country == "{country}"')['Sample'].to_numpy()
    
    # global_progressbar.update(f'Analysing {len(samples_ids)} samples')
    
    primers_variation = [
        streamlit_assess_variation_in_sequence(
            primers_info[p],
            samples_ids,
            callset,
            only_pass_variants=False
        )
        for p in primers_info.keys()
    ]
    
    primer_variation_df = pd.concat(
        primers_variation,
        axis=0,
        ignore_index=True
    )

    significance_flags = (
            (primer_variation_df.freq >= significant_variation_frequency) &
            (primer_variation_df.seq_allele != primer_variation_df.allele) &
            (primer_variation_df.type != 'reference')
    )
    primer_variation_df['is_significant'] = significance_flags
    
    # global_progressbar.finish()
    
    return primer_variation_df, primer_coords_df
    
#     st.download_button("Download primer variation CSV file",
#                        primer_variation_df.to_csv(index = False).encode('utf-8'),
#                        "primer-variation.csv")
    
#     st.download_button("Download primer coordinates CSV file",
#                        primer_coords_df.to_csv(index = False).encode('utf-8'),
#                        "primer-coordinates.csv")

# class progress_bar_wrapper:
#     def __init__(self):
#         self.progress = 0
#         self.message = ""
#         self.progressbar = st.progress(self.progress, self.message)
    
#     def update(self, new_message):
#         self.progress += 3
#         self.message += f"{new_message}... "
        
#         self.progressbar.progress(self.progress, self.message)
    
#     def finish(self):
#         self.message += f"and done!"
#         self.progressbar.progress(100, self.message)
        
#         time.sleep(0.2)
#         self.progressbar.empty()

# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== 
# STREAMLIT APP
# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== 

st.title("Primer Checker")

# LOAD VARIABLES

pf7, metadata, callset = cache_pf7_resources()

# FILE UPLOAD

uploaded_csv = st.file_uploader("Step 1: Upload primer data .csv file", type = "csv")

if uploaded_csv == None:
    st.stop()
    pass

primer_data = cache_read_csv(uploaded_csv)
with st.expander("Click to see the file you've just uploaded"):
    st.dataframe(primer_data)

# COUNTRY SELECT

countries = sorted(metadata.Country.dropna().unique())

selected_country = st.selectbox("Step 2: Choose country", options = ["--", "All countries"] + countries)

if selected_country == "--":
    st.stop()

# SIGNIFICANCE SELECT

selected_significance_frequency = st.number_input("Step 3: Choose significance frequency", value = 0.05)

# RUN WORKFLOW

# global global_progressbar
# global_progressbar = progress_bar_wrapper()

primer_variation_df, primer_coords_df = streamlit_run_workflow_pf7(selected_country,
                                                                   selected_significance_frequency)

download_col1, download_col2 = st.columns(2)
expander_col1, expander_col2 = st.columns(2)

download_col1.download_button("Click to download primer variation CSV file",
                              primer_variation_df.to_csv(index = False).encode('utf-8'),
                              "primer-variation.csv",
                              use_container_width = True
                             )

download_col2.download_button("Click to download primer coordinates CSV file",
                              primer_coords_df.to_csv(index = False).encode('utf-8'),
                              "primer-coordinates.csv",
                              use_container_width = True
                             )

with expander_col1.expander("Click to check out the primer variation dataframe you've just created"):
    st.dataframe(primer_variation_df)
    
with expander_col2.expander("Click to check out the primer coordinates dataframe you've just created"):
    st.dataframe(primer_coords_df)
