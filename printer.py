import pickle

with open('./Data/char_to_idx.pickle', 'rb') as handle:
    char_to_idx = pickle.load(handle)


def intToSelfies(encoded):
    encoded = encoded.tolist()
    encoded_rounded = [int(round(i)) if i < 110.5 else 110 for i in encoded]
    
    count = 0 
    for i in range(len(encoded_rounded)):  # to remove redundant values at the end
        if encoded_rounded[i] > 106:
            count += 1
        if count == 3:
            encoded_rounded = encoded_rounded[: i - 2]
            break
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    decoded = [idx_to_char[i] for i in encoded_rounded]
    decoded = [elem for elem in decoded if elem != '']
    return decoded