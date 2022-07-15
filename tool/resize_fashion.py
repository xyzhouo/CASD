import os
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES=True
def resize_dataset(folder, new_folder, new_size = (256, 256), crop_bord=0):
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    for name in os.listdir(folder):
        old_name = os.path.join(folder, name)
        new_name = os.path.join(new_folder, name)


        img = Image.open(old_name)
        w, h =img.size
        if crop_bord == 0:
            pass
        else:
            img = img.crop((crop_bord, 0, w-crop_bord, h))
        img = img.resize([new_size[1],new_size[0]])
        img.save(new_name)
        print('resize %s succefully' % old_name)


old_dir = '../dataset/fashion/train'
root_dir = '../dataset/fashion/train_resize'
resize_dataset(old_dir, root_dir)

