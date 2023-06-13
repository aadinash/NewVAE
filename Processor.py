# Tasks from basic: add splits, print actual smiles, add latent space knowledge

import selfies
from sklearn.model_selection import train_test_split
import pickle
import torch

x_values = []

with open('CS197_dataset/DATASET.csv', 'r') as f:
    lines = f.readlines()

with open('CS197_dataset/ESM_embedding.pickle', 'rb') as fd:
    data = pickle.load(fd)

# overrides default constraints
selfies.set_semantic_constraints({'?': 8})

selfies_list = []

for item in lines:
    A = item.split(',')
    encoded = selfies.encoder(A[0])
    selfies_list.append(encoded)

all_selfies_symbols = selfies.get_alphabet_from_selfies(selfies_list)
selfies_alphabet = list(all_selfies_symbols)
largest_selfies_len = max(selfies.len_selfies(s) for s in selfies_list)

### borrowed
pad_token = ''
char_to_idx = {x:i for i,x in enumerate(selfies_alphabet)}
char_to_idx[pad_token] = len(char_to_idx)
###

print(char_to_idx)

for i, item in enumerate(selfies_list):
    ### borrowed
    selfies_chr = item.split('[')
    selfies_chr = ['['+x for x in selfies_chr if x != '']
    encoded = [char_to_idx[char.rstrip('.')] for char in selfies_chr]
    encoded += [char_to_idx[pad_token]] * (largest_selfies_len - len(encoded))
    encoded = [x/len(char_to_idx) for x in encoded] # normalize to between 0-1
    encoded_tensor = torch.tensor(encoded, dtype=torch.float32)
    ###

    # We formerly appended tuples. Now, we will just append a concatenation
    concat = torch.cat((encoded_tensor, torch.sigmoid(data[i])))
    x_values.append(concat)

X_train, X_test = train_test_split(x_values, test_size=0.15, random_state=42)

train_combined_repr_tensor = torch.stack(X_train)
print(train_combined_repr_tensor)
# Save the tensor
torch.save(train_combined_repr_tensor, './Data/train_combined_repr_tensor.pt')

test_combined_repr_tensor = torch.stack(X_test)
torch.save(train_combined_repr_tensor, './Data/test_combined_repr_tensor.pt')

# Save the char_to_idx mapping
with open('./Data/char_to_idx.pickle', 'wb') as handle:
    pickle.dump(char_to_idx, handle)
