import torch


def shuffle_array(array, p=0.15):
    """Randomly shuffle a percentage of the elements in an array."""

    # Generate a binary mask for the element to be shuffled
    mask = torch.bernoulli(torch.ones(array.shape) * p).to(array.dtype)

    # Shuffle the elements that are assigned to the binary mask
    shuffled = array[mask == 1][torch.randperm(array[mask == 1].nelement())]

    # Create a dummy array that contains the newly shuffled elements
    indices = torch.zeros(array.shape, dtype=array.dtype)
    indices[array[mask == 1]] = shuffled

    # Fill the original array with the shuffled elements
    result = torch.where(mask == 1, indices, array)

    return result
