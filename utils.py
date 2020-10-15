import megengine as mge
from megengine.module import Module
import megengine.functional as F
class Swish(Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(np.ones(1, dtype=np.float32))

    def forward(self, inp):
        return F.sigmoid(inp * self.weight) * inp

def frame_count(container, video_stream=0):
    def count(generator):
        res = 0
        for _ in generator:
            res += 1
        return res
        
    frames = container.streams.video[video_stream].frames
    if frames != 0:
        return frames
    frame_series = container.decode(video=video_stream)
    frames = count(frame_series)
    container.seek(0)
    return frames

def tsm(x):
    # tensor [N*T, C, H, W]
    size = x.shape
    tensor = x.reshape((-1, 5) + size[1:])
    # tensor [N, T, C, H, W]
    p = size[1] // 4
    pre_tensor = tensor[:, :, :p]
    post_tensor = tensor[:, :, p:2*p]
    peri_tensor = tensor[:, :, 2*p:]
    pre_tensor_  = F.concat((mge.zeros(pre_tensor[:, -1: ], dtype=tensor.dtype),
                            pre_tensor [:,   :-1]), 1)
    post_tensor_ = F.concat((post_tensor[:,  1: ],
                             mge.zeros(post_tensor[:,   :1], dtype=tensor.dtype)), 1)
    output = F.concat((pre_tensor_, post_tensor_, peri_tensor), 2).reshape(size)
    output = tensor.reshape(size)
    return output
