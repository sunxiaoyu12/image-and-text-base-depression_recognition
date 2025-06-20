import os
import cv2
import json
import fnmatch
import shutil
import pdb

def image_count(dir):
    # images_num = fnmatch.filter(os.listdir(dir), "*.jpg")
    images_num = len(os.listdir(dir)) -1
    return images_num

def copy_images(imagedir, imagename, tartget_dir):
    im = cv2.imread(os.path.join(imagedir, imagename))
    try:
        cv2.imwrite(os.path.join(tartget_dir, imagename), im)
        # shutil.copyfile(img_path, tartget_dir)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    # train.txt : image path\tcontent\tlabel
    data_path = r"D:/datasets/MultiModalDataset/positive/"
    target_path = r"D:/datasets/Mul_1/"

    for dirs_path, dirs_names, files in os.walk(data_path):
        for dir in dirs_names:
            full_path = os.path.join(dirs_path, dir)  # user name
            # print(full_path)
            # len_images = image_count(full_path)
            # print("len_images:",len_images)
            txt_path = os.path.join(full_path, "timeline.txt")  # path to timeline.txt
            # print("txt_path:",txt_path)
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as file:
                    # len_txt = len(file.readlines())
                    # print("len_txt*******", len_txt)

                    # if int(len_images) > 0:
                    #     print(len_images > 0)
                    #     pdb.set_trace()
                    for line in file.readlines():
                        print("line----->", line)  # "dict"
                        dic = json.loads(line)
                        id_str = dic["id_str"]
                        img_path = os.path.join(full_path, id_str + ".jpg")
                        print("img_path------------", img_path)
                        text = dic["text"]
                        print("text========", text)
                        if os.path.exists(img_path) and (text is not None):
                            copy_images(full_path, id_str + ".jpg", target_path)
                            # print("target txt path:", target_path + id_str + ".txt")
                            with open(target_path + id_str + ".txt", "w", encoding='utf-8') as f:
                                f.write(text)
                            f.close()
                file.close()

