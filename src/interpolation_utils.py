import torch


def interpolate(latents, latent_indices, interpolation_type):

    interpolate = None
    if interpolation_type == 'linear':
        interpolate = lerp
    elif interpolation_type == 'slerp':
        interpolate = slerp
    else:
        raise Exception('Wrong interpolation type specified')

    latents_generated = []
    for i in range(1, len(latents)):
        left_latent, right_latent = latents[i-1], latents[i]
        left_idx, right_idx = latent_indices[i-1], latent_indices[i]
        latents_generated.append(left_latent)

        step_size = 1 / (right_idx - left_idx)
        for step in range(1, right_idx - left_idx):
            latent_generated = interpolate(step * step_size, left_latent, right_latent)
            latents_generated.append(latent_generated)

    latents_generated.append(latents[-1])
    latents_generated = torch.stack(latents_generated, dim=0)
    return latents_generated


def lerp(val, low, high):
    return low + val * (high - low)


def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res
