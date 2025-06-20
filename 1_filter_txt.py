import os
import fnmatch
import random
import shutil
import tqdm


# 删除没有对应image的txt文件
# txt_list = fnmatch.filter(os.listdir(positive_path), "*.txt")
# for txt_file in txt_list:
#     idx = txt_file.split(".")[0]
#     image_name = idx + ".jpg"
#     if not os.path.exists(os.path.join(positive_path, image_name)):
#         os.remove(os.path.join(positive_path, txt_file))

# split txt file into train and test
nagative_path = r"D:/datasets/Mul_0/"
def split_txt_file(positive_path, nagative_path, target_path, train_ratio=0.8):
    random.seed(99)  # For reproducibility
    pos_list = fnmatch.filter(os.listdir(positive_path), "*.txt")
    total_pos_files = len(pos_list)
    pos_train_count = int(total_pos_files * train_ratio)
    random.shuffle(pos_list)  # Shuffle the list to ensure randomness
    # train_pos = pos_list[:pos_train_count]
    test_pos = pos_list[pos_train_count:]
    
    nag_list = fnmatch.filter(os.listdir(nagative_path), "*.txt")
    total_nag_files = len(nag_list)
    nag_train_count = int(total_nag_files * train_ratio)
    random.shuffle(nag_list)  # Shuffle the list to ensure randomness
    # train_nag = nag_list[:nag_train_count]
    test_nag = nag_list[nag_train_count:]
    
    label = [0, 1]  # 0 for negative, 1 for positive
    # with open(os.path.join(target_path, "train.txt"), "w", encoding='utf-8') as train_f:
    #     for file in tqdm.tqdm(train_pos):  # file is the name of the txt file
    #         img_path = os.path.join(positive_path, file.replace(".txt", ".jpg"))
    #         with open(os.path.join(positive_path, file), "r", encoding='utf-8') as f:
    #             content = f.read()
    #             train_f.write(f"{img_path}\t{content}\t{label[1]}\n")
    #             shutil.copy(os.path.join(positive_path, file), os.path.join(target_path, file))
    #             shutil.copy(img_path, os.path.join(target_path, file.replace(".txt", ".jpg")))
    #     for file in tqdm.tqdm(train_nag):  # file is the name of the txt file
    #         img_path = os.path.join(nagative_path, file.replace(".txt", ".jpg"))
    #         with open(os.path.join(nagative_path, file), "r", encoding='utf-8') as f:
    #             content = f.read()
    #             train_f.write(f"{img_path}\t{content}\t{label[0]}\n")
    #             shutil.copy(os.path.join(nagative_path, file), os.path.join(target_path, file))
    #             shutil.copy(img_path, os.path.join(target_path, file.replace(".txt", ".jpg")))
        
    # train_f.close()

    with open(os.path.join(target_path, "test.txt"), "w", encoding='utf-8') as test_f:
        for file in tqdm.tqdm(test_pos):
            img_path = os.path.join(positive_path, file.replace(".txt", ".jpg"))
            with open(os.path.join(positive_path, file), "r", encoding='utf-8') as f:
                content = f.read()
                test_f.write(f"{img_path}\t{content}\t{label[1]}\n")
                shutil.copy(os.path.join(positive_path, file), os.path.join(target_path, file))
                shutil.copy(img_path, os.path.join(target_path, file.replace(".txt", ".jpg")))
        for file in tqdm.tqdm(test_nag):
            img_path = os.path.join(nagative_path, file.replace(".txt", ".jpg"))
            with open(os.path.join(nagative_path, file), "r", encoding='utf-8') as f:
                content = f.read()
                test_f.write(f"{img_path}\t{content}\t{label[0]}\n")
                shutil.copy(os.path.join(nagative_path, file), os.path.join(target_path, file))
                shutil.copy(img_path, os.path.join(target_path, file.replace(".txt", ".jpg")))
    test_f.close()
    

if __name__ == "__main__":
    
    positive_path = r"D:/datasets/Mul_1/"
    nagative_path = r"D:/datasets/Mul_0/"
    target_path = r"D:/datasets/Mul_all/"
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    split_txt_file(positive_path, nagative_path, target_path, train_ratio=0.8)
  