import matplotlib.pyplot as plt  # noqa
import numpy as np
import torch


def subsequent_mask(size):
    attn_shape = (1, size, size)

    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")

    return torch.from_numpy(1 - subsequent_mask)


# print(subsequent_mask(5))
# plt.figure(figsize=(5, 5))
# plt.imshow(subsequent_mask(20)[0])
# plt.savefig("visualization_subsequent_mask.png")
