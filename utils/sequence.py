import numpy as np

def search_sequence_numpy(arr: np.ndarray[np.int64], seq: np.ndarray[np.int64]) -> np.ndarray[np.int64]:
    """
    Search for the occurrence of a sequence within a numpy array.

    Parameters:
    arr: np.ndarray[np.int64] - the input array in which the sequence will be searched
    seq: np.ndarray[np.int64] - the sequence to search for within the input array

    Returns:
    np.ndarray[np.int64] - a numpy array of indices where the sequence is found in the input array.
              If no match is found, an empty list is returned.
    """
    arr_length, seq_length = arr.size, seq.size

    # Create an array of indices for the sequence.
    seq_indices = np.arange(seq_length)

    # Compare the elements in 'arr' with 'seq' to find matching subsequences.
    is_match = (arr[np.arange(arr_length - seq_length + 1)[:, None] + seq_indices] == seq).all(1)
    
    # If any match is found, return the indices where the sequence is found.
    if is_match.any():
        # Use convolution to find the starting indices of matching sequences.
        return np.where(np.convolve(is_match, np.ones(seq_length, dtype=int) > 0))[0]
    return np.array([], dtype=np.int64)  # No match found
