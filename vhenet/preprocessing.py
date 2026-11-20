"""Data preprocessing for FASTA sequences, interaction tables, and host similarity."""

import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, List
import itertools
from pathlib import Path
import sys
import warnings
# Make the repository root importable.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class KmerVocab:
    """K-mer vocabulary helper."""

    def __init__(self, k: int = 3):
        self.k = k
        self.bases = ['A', 'C', 'G', 'T']
        self.vocab = self._build_vocab()

    def _build_vocab(self) -> Dict[str, int]:
        """Build the k-mer vocabulary; index 0 is reserved for padding."""
        kmers = []
        for item in itertools.product(self.bases, repeat=self.k):
            kmers.append("".join(item))
        vocab = {kmer: i + 1 for i, kmer in enumerate(kmers)}  # 0 for padding
        return vocab

    def __len__(self):
        return len(self.vocab) + 1  # +1 for padding token

    def __getitem__(self, kmer: str) -> int:
        return self.vocab.get(kmer, 0)

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab.copy()


def parse_fasta(fasta_file: str) -> Dict[str, str]:
    """
    Parse a FASTA file and return a {virus_name: sequence} dictionary.

    Args:
        fasta_file: Path to the FASTA file.

    Returns:
        Dictionary mapping virus names to sequences.
    """
    seq_dict = {}
    try:
        from Bio import SeqIO

        for record in SeqIO.parse(fasta_file, "fasta"):
            # The virus name is the part of the header before "Accessions:".
            if 'Accessions:' in record.description:
                # Standard header format.
                virus_name = record.description.split('Accessions:')[0].strip()
            else:
                # Legacy headers use a pipe separator.
                virus_name = record.description.split('|')[0].strip()
            seq = str(record.seq).upper()
            seq_dict[virus_name] = seq
    except FileNotFoundError:
        print(f"Error :  FASTA file not found at {fasta_file}")
        sys.exit(1)
    except Exception:
        name = None
        parts = []
        try:
            with open(fasta_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('>'):
                        if name is not None:
                            seq_dict[name] = ''.join(parts).upper()
                        desc = line[1:]
                        if 'Accessions:' in desc:
                            name = desc.split('Accessions:')[0].strip()
                        else:
                            name = desc.split('|')[0].strip()
                        parts = []
                    else:
                        parts.append(line)
                if name is not None:
                    seq_dict[name] = ''.join(parts).upper()
        except FileNotFoundError:
            print(f"Error :  FASTA file not found at {fasta_file}")
            sys.exit(1)
    return seq_dict


def seq_to_ids(seq: str, vocab: KmerVocab, max_len: int = 100) -> torch.Tensor:
    """
    Convert a sequence into a k-mer id sequence.

    Args:
        seq: DNA sequence.
        vocab: KmerVocab instance.
        max_len: Maximum sequence length.

    Returns:
        Id tensor with shape (max_len,).
    """
    ids = []
    for i in range(len(seq) - vocab.k + 1):
        kmer = seq[i:i + vocab.k]
        ids.append(vocab[kmer])

    # Pad or truncate to the requested length.
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))  # padding with 0
    else:
        ids = ids[:max_len]

    return torch.tensor(ids, dtype=torch.long)


def seq_to_ids_sliding_window(
        seq: str,
        vocab: KmerVocab,
        max_kmers: int = 512,  # Number of k-mers.
        window_kmers: int = 256,  # K-mers per window.
        stride_kmers: int = 128,  # Stride in k-mers.
        mode: str = 'train'
) -> torch.Tensor:
    """Split a long sequence into windows measured in k-mers.

    Args:
        seq: DNA sequence.
        vocab: KmerVocab instance.
        max_kmers: Maximum k-mers per fragment.
        window_kmers: K-mers per window.
        stride_kmers: Stride in k-mers.
        mode: 'train' or 'val'.

    Returns:
        Fragment-id tensor with shape (num_fragments, max_kmers).
    """
    k = vocab.k
    seq_len = len(seq)

    # Number of k-mers in the whole sequence.
    total_kmers = max(0, seq_len - k + 1)

    # Short sequence: one fragment.
    if total_kmers <= window_kmers:
        # Single fragment [1, max_kmers].
        single_fragment = seq_to_ids_simple(seq, vocab, max_kmers).unsqueeze(0)
        # Pad to ten fragments with zero rows.
        padding = torch.zeros(
            (10 - 1, max_kmers),
            dtype=torch.long
        )
        # Concatenated shape: [10, max_kmers].
        full_fragments = torch.cat([single_fragment, padding], dim=0)
        return full_fragments

    # Build all k-mer ids.
    all_kmer_ids = []
    for i in range(total_kmers):
        kmer = seq[i:i + k]
        all_kmer_ids.append(vocab[kmer])

    # Sliding-window fragments.
    fragments = []
    for start in range(0, total_kmers - window_kmers + 1, stride_kmers):
        end = start + window_kmers
        fragment_ids = all_kmer_ids[start:end]

        # Pad or truncate each fragment to max_kmers.
        if len(fragment_ids) < max_kmers:
            fragment_ids = fragment_ids + [0] * (max_kmers - len(fragment_ids))
        else:
            fragment_ids = fragment_ids[:max_kmers]

        fragments.append(torch.tensor(fragment_ids, dtype=torch.long))

        # Cap the fragment count to bound memory use.
        if len(fragments) >= 4:
            break

    if not fragments:
        # Fall back to the whole sequence if no window was produced.
        return seq_to_ids_simple(seq, vocab, max_kmers).unsqueeze(0)

    num_fragments = len(fragments)
    if num_fragments < 10:
        # Zero padding for the missing fragments.
        padding_fragments = torch.zeros(
            (10 - num_fragments, max_kmers),
            dtype=torch.long
        )
        # Append the padding rows.
        fragments = fragments + [padding_fragments[i] for i in range(padding_fragments.shape[0])]

    return torch.stack(fragments, dim=0)


def seq_to_ids_simple(seq: str, vocab: KmerVocab, max_kmers: int = 1000) -> torch.Tensor:
    """Convert one sequence into a fixed-length k-mer id vector.

    Args:
        seq: DNA sequence.
        vocab: KmerVocab instance.
        max_kmers: Maximum k-mers.

    Returns:
        K-mer id tensor with shape (max_kmers,).
    """
    k = vocab.k
    seq_len = len(seq)

    # Extract all k-mers.
    ids = []
    for i in range(max(0, seq_len - k + 1)):
        kmer = seq[i:i + k]
        ids.append(vocab[kmer])

    # Pad or truncate to a fixed length.
    if len(ids) < max_kmers:
        ids += [0] * (max_kmers - len(ids))
    else:
        ids = ids[:max_kmers]

    return torch.tensor(ids, dtype=torch.long)


def process_interaction_data(
        interaction_csv: str,
        similarity_csv: str,
        host_col: str = "V-H"
) -> Tuple[pd.DataFrame, Dict[str, int], torch.Tensor]:
    """Load the interaction table and host similarity matrix.

    Args:
        interaction_csv: Path to the virus-host interaction table.
        similarity_csv: Path to the host similarity matrix.
        host_col: Column containing "Virus__Host" identifiers.

    Returns:
        (interaction DataFrame, host-to-id mapping, similarity tensor)
    """
    # Load the interaction table.
    interaction_path = Path(interaction_csv)
    suffix = interaction_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(interaction_path)
    elif suffix in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(interaction_path)
        except ValueError as exc:
            # Pandas raises this when a text/CSV file has an Excel extension.
            with interaction_path.open("rb") as file_obj:
                header = file_obj.read(256)
            if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(
                    f"{interaction_path} is a Git LFS pointer, not an Excel workbook. "
                    "Run 'git lfs pull' to download the data file."
                ) from exc
            if b"\x00" in header:
                raise ValueError(
                    f"{interaction_path} is not a readable Excel workbook. "
                    "Replace the file or point data.interaction_csv to the CSV copy."
                ) from exc
            warnings.warn(
                f"{interaction_path} has an Excel extension but contains text; "
                "loading it as CSV.",
                RuntimeWarning,
                stacklevel=2,
            )
            df = pd.read_csv(interaction_path)
    else:
        raise ValueError(
            f"Unsupported interaction data format '{suffix or '<none>'}' for "
            f"{interaction_path}; expected .csv, .xlsx, or .xls."
        )

    # Split the V-H column into Virus and Host names.
    if host_col not in df.columns:
        raise KeyError(
            f"Required column '{host_col}' is missing from {interaction_path}. "
            f"Available columns: {list(df.columns)}"
        )
    df[['Virus', 'Host']] = df[host_col].str.split('__', expand=True)

    # Unique hosts in sorted order.
    unique_hosts = sorted(df['Host'].unique())
    host_to_id = {h: i for i, h in enumerate(unique_hosts)}

    # Load the similarity matrix.
    sim_df = pd.read_csv(similarity_csv, index_col=0)

    # Reorder the similarity matrix to match host_to_id.
    ordered_hosts = sorted(host_to_id.keys())
    sim_matrix = sim_df.loc[ordered_hosts, ordered_hosts].values
    sim_matrix = torch.tensor(sim_matrix, dtype=torch.float32)

    return df, host_to_id, sim_matrix


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load the YAML configuration.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Configuration dictionary.
    """
    import yaml

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        current_dir = Path("")
        print(f"Current directory: {current_dir.absolute()}")
        print(f"⚠️  Config file not found at {config_path}, using defaults")
        return get_default_config()


def get_default_config() -> Dict:
    """Return a minimal default configuration."""
    return {
        'embed_dim': 32,
        'seq_max_len': 60,
        'kmer': 3,
        'batch_size': 16,
        'epochs': 50,
        'learning_rate': 1e-3,
        'device': 'cuda',
        'train_test_split': 0.8,
    }


# ============================================
# Data validation and statistics
# ============================================

def validate_data(df: pd.DataFrame, seq_dict: Dict, host_to_id: Dict) -> Dict:
    """Validate data completeness.

    Args:
        df: Interaction table.
        seq_dict: Virus sequence dictionary.
        host_to_id: Host-to-id mapping.

    Returns:
        Validation report.
    """
    report = {
        'num_interactions': len(df),
        'num_viruses': df['Virus'].nunique(),
        'num_hosts': df['Host'].nunique(),
        'num_sequences': len(seq_dict),
        'missing_sequences': 0,
        'missing_hosts': 0,
        'pos_samples': (df['Label'] == 1).sum(),
        'neg_samples': (df['Label'] == 0).sum(),
    }

    # Check for missing sequences.
    for virus in df['Virus'].unique():
        if virus not in seq_dict:
            report['missing_sequences'] += 1

    # Check for missing hosts.
    for host in df['Host'].unique():
        if host not in host_to_id:
            report['missing_hosts'] += 1

    return report


def print_data_stats(report: Dict):
    """Print the data statistics report."""
    print("\n" + "=" * 50)
    print("Data statistics")
    print("=" * 50)
    print(f"interaction pairs: {report['num_interactions']}")
    print(f"viruses:           {report['num_viruses']}")
    print(f"hosts:             {report['num_hosts']}")
    print(f"sequences:         {report['num_sequences']}")
    print(f"missing sequences: {report['missing_sequences']}")
    print(f"missing hosts:     {report['missing_hosts']}")
    print(f"positives:         {report['pos_samples']}")
    print(f"negatives:         {report['neg_samples']}")
    print(
        f"positive/negative: {report['pos_samples'] / report['neg_samples']:.3f}" if report['neg_samples'] > 0 else "N/A")
    print("=" * 50 + "\n")
