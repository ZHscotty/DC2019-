import os
from PIL import Image


path = 'E:\比赛数据\新网银行唇语识别竞赛数据\\1.训练集\lip_train'
save_path = 'E:\比赛数据\新网银行唇语识别竞赛数据\select'
files = os.listdir(path)[:1000]
for x in range(len(files)):
    file_path = os.path.join(path, files[x])
    Imgs = os.listdir(file_path)[0]
    Img_path = os.path.join(file_path, Imgs)
    im = Image.open(Img_path)
    im.save(os.path.join(path, '%.4d.jpg'%(x+1)))
    print('save image {}'.format(x))
