import os
from PIL import Image
from torchvision import transforms as tvt
from torchvision.utils import save_image
import torch


class FrameIO:

    def __init__(self, frames_path, skip=4, frame_dims=(640, 480),
                 device='cuda' if torch.cuda.is_available() else 'cpu') -> None:
        self.frames_path = frames_path
        self.skip = skip
        self.frame_dims = frame_dims
        self.img_to_tensor = tvt.ToTensor()
        self.filenames = sorted([f for f in os.listdir(self.frames_path) if f.endswith('.jpg')])
        self.device = device
        self.dtype = torch.float16
        self.frames = None
        self.frame_indices = None
        self.frame_dims_orig = None

    def load(self):
        self.frames = []
        self.frame_indices = []

        idx = -1
        for filename in self.filenames:
            idx += 1
            if idx % (self.skip + 1) != 0 and idx != len(self.filenames) - 1:
                continue

            frame_path = os.path.join(self.frames_path, filename)
            pil_frame_orig = Image.open(frame_path).convert('RGB')
            self.frame_dims_orig = pil_frame_orig.size

            pil_frame = pil_frame_orig.resize(self.frame_dims, Image.Resampling.LANCZOS)
            x = self.img_to_tensor(pil_frame)

            self.frames.append(x)
            self.frame_indices.append(idx)

        self.frames = torch.stack(self.frames, dim=0).to(device=self.device, dtype=self.dtype)
        return self

    def save(self, frames_generated, folder_suffix):
        if len(frames_generated) != len(self.filenames):
            raise Exception('Length of the frames that generated must be equal to the size of the original directory')

        frames_path_split = self.frames_path.split('/')
        frames_generated_path = ('/'.join(frames_path_split[:-2]) + '/' + frames_path_split[-2] +
                                 '_' + folder_suffix + '/' + frames_path_split[-1])

        os.makedirs(frames_generated_path, exist_ok=True)
        for i, frame_generated in enumerate(frames_generated):
            frame_generated = frame_generated.resize(self.frame_dims_orig, Image.Resampling.LANCZOS)
            frame_generated.save(os.path.join(frames_generated_path, self.filenames[i]))






