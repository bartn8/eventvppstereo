import torch

def sample_hints(hints, validhints, probability=0.15):
    if probability < 1:
        new_validhints = (validhints * (torch.rand_like(validhints, dtype=torch.float32) < probability)).float()
        new_hints = hints * new_validhints  # zero invalid hints
        new_hints[new_validhints==0] = 0
    else:
        new_hints = hints
        new_validhints = validhints

    return new_hints, new_validhints