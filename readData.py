import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


# # 定义训练集增强算子
# train_transforms = T.Compose([
#     T.RandomCrop(crop_size=224),
#     T.RandomHorizontalFlip(),
#     T.Normalize()])

#定义读取文件的格式
def default_loader(path):
    img = Image.open(path).convert('RGB')
    return img

class MyDataset(Dataset):
    def __init__(self, data_txt, transform=None,loader=default_loader):
        self.transform = transform
        self.loader = loader
        fh = open(data_txt, 'r')  # 按照传入的路径和txt文本参数以只读的方式打开这个文本
        all_data = []
        for line in fh.readlines():
            
            line = line.strip().split(',')
            # print(line)
            # import pdb; pdb.set_trace()
            if len(line) < 3:
                continue
            image_path = line[0]
            content_path = line[1]
            label = int(line[2])
            
            all_data.append((image_path, content_path, label))
        fh.close()
        self.all_data = all_data
        

    def __getitem__(self, index):
        img_path, text_path, label = self.all_data[index]
        image = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(image)
        else:
            img = np.array(image)
        text = open(text_path, 'r', encoding='utf-8').read().strip()
        return img, text, label

    def __len__(self):
        return len(self.all_data)
    

if __name__ == "__main__":
    data_transform = {
    "train": transforms.Compose([transforms.Resize([256,256]),
                                 transforms.RandomRotation(10),
                                 transforms.RandomResizedCrop(224,scale=(0.8, 1.0), ratio=(0.95, 1.05)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5], [0.5])]),#来自官网参数
    "val": transforms.Compose([transforms.Resize([224,224]),#将最小边长缩放到256
                               transforms.ToTensor(),
                               transforms.Normalize([0.5], [0.5])])}
    
    train_datasets = MyDataset("/nfs/xy_outputs/depression/test.txt", transform=data_transform["val"])
    traindataloader = DataLoader(train_datasets, batch_size=16, shuffle=True, num_workers=0)
    for i, traindata in enumerate(traindataloader):
        print('i:',i)
        img, content, Label = traindata
        print("img:", img)
        print('data:', content)
        print('Label:', Label)