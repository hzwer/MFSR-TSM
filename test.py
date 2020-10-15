from model import *
from IPython.display import Image, display
import tarfile
import cv2
import numpy as np
import os

MODEL_PATH = "model.mge.state"
TEST_RAW_DATA = "../../../dataset/game1/test.tar"

net = SimpleUNet()

with open(MODEL_PATH, 'rb') as f:
    net.load_state_dict(mge.load(f)['net'])
    
@mge.jit.trace
def inference(inp):
    return net(inp)

for i in range(90, 100):
    cur_dir = '/home/megstudio/workspace/input/{}/'.format(i)
    save_dir = '/home/megstudio/workspace/test/{}/'.format(i)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    l = os.listdir(cur_dir)
    for j in l:
        if 'png' in j:
            img_dir = cur_dir + j
            img = cv2.imread(img_dir)
            img = cv2.resize(img, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            img = np.float32(img / 255.)
            h, w, _ = img.shape
            ph = ((h - 1) // 64 + 1) * 64
            pw = ((w - 1) // 64 + 1) * 64
            img = np.pad(img, ((0,ph-h),(0,pw-w),(0,0)), 'constant')
            print(img_dir, img.shape)
            img = img.transpose((2, 0, 1))[None, :, :, :]
            img_out = inference(img).numpy()[0]
            print(np.abs(img - img_out).mean())
            img_out = ((img_out * 255).clip(0, 255).transpose((1, 2, 0)).copy()[:h, :w]).astype('uint8')
            cv2.imwrite(save_dir + j, img_out)

'''
with tarfile.open(TEST_RAW_DATA, mode='r') as tar:
    tinfo = tar.getmember("test/90/0045.png")
    content = tar.extractfile(tinfo).read()
    img = cv2.imdecode(np.frombuffer(content, dtype='uint8'), 1)
    img = cv2.resize(img, (0, 0), fx=4, fy=4)
    img = (np.float32(img) / 256).transpose((2, 0, 1))[None, :, :, :]
    img_out = inference(img)
    img_out = (img_out.numpy() * 256).clip(0, 255)[0].transpose((1, 2, 0)).copy()
    content_out = cv2.imencode('.png', img_out)[1]
    
    display(Image(data=content_out, width=400))
'''
