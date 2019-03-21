import random


def snapshot(*, time, seq, index, value=0):
    """
    Prints sequence, index and value at a given time

    Args:
        time: time in seconds
        seq: sequence
        index: sample index of the given sequence
        value: the value of the sequence at index
    """
    print((f"time {time:02}, actual sequence is {seq}, "
           f"sample index {index}, returned {value};"))


def generate_new_seq(seq=None, seq_length=5):
    """
    Generates a new sequence

    Args:
        seq: sequence of zeroes and one one
        seq_length: length of sequence

    Returns:
        New sequence of length seq_length.
    """
    bin_format = '005b'
    if seq is None:
        position = random.randint(0, seq_length - 1)
        return format(2 ** position, bin_format)

    move_right = random.randint(0, 1)
    if int(seq[0]) == 1 or move_right: # very right or move right
        seq = format(int(seq, 2) >> 1, bin_format)
    else: # very left or move left
        seq = format(int(seq, 2) << 1, bin_format)

    return seq


def find_index(*, time, seq):
    """
    Finds the index of the value 1 in the sequence.

    Args:
        time: time in seconds
        seq: sequence of zeroes and one one

    Returns:
        Index of the value 1. -1 if 1 does not exist in the sequence.
    """
    seq_length = len(seq)
    current_index = seq_length - 1

    while current_index > 0:
        wait_count = current_index + 1
        while wait_count > 0:
            if int(seq[current_index]) == 1:
                snapshot(time=time, seq=seq, index=current_index, value=1)
                return current_index
            wait_count -= 1
            snapshot(time=time, seq=seq, index=current_index)
            seq = generate_new_seq(seq)
            time += 1
        current_index -= 1
    return -1


if __name__ == '__main__':
    sequence = generate_new_seq()
    find_index(time=0, seq=sequence)
