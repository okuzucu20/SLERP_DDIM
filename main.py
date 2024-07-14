import torch.cuda

from src.ddim_inversion import DDIMInversionPipeline
from src.frame_io import FrameIO
from src.interpolation_utils import interpolate
import os
import sys

def main():

    if len(sys.argv) < 2:
        raise Exception('Data directory must be specified!')

    data_path = sys.argv[1]
    skip = 4

    ddim_pipe = DDIMInversionPipeline()

    for frames_dir in os.listdir(data_path):

        print(f"Generating interpolated frames using directory: {frames_dir}")

        ### Create frame tensors using FrameIO ###
        frames_path = os.path.join(data_path, frames_dir)
        frame_io = FrameIO(frames_path, skip=skip).load()

        ### Calculate ddim latents ###
        inv_latents = ddim_pipe.encode_and_diffuse(frame_io.frames)
        inv_latents_indices = frame_io.frame_indices

        ### Interpolate ###
        inv_latents_interp_linear = interpolate(inv_latents, inv_latents_indices, 'linear')
        inv_latents_interp_slerp = interpolate(inv_latents, inv_latents_indices, 'slerp')

        ### Restore interpolated frames ###
        frames_generated_linear = ddim_pipe.denoise_and_decode(inv_latents_interp_linear)
        frames_generated_slerp = ddim_pipe.denoise_and_decode(inv_latents_interp_slerp)

        ### Save generated images ###
        frame_io.save(frames_generated_linear, 'linear')
        frame_io.save(frames_generated_slerp, 'slerp')

    for frames_dir in os.listdir(data_path):

        print(f"Generating reconstructed frames using directory: {frames_dir}")

        ### Create frame tensors using FrameIO ###
        frames_path = os.path.join(data_path, frames_dir)
        frame_io = FrameIO(frames_path, skip=0).load()

        ### Reconstruct ###
        inv_latents = ddim_pipe.encode_and_diffuse(frame_io.frames)
        frames_generated_recon = ddim_pipe.denoise_and_decode(inv_latents)

        ### Save generated images ###
        frame_io.save(frames_generated_recon, 'recon')




if __name__ == '__main__':
    main()


