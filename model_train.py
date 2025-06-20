import cv2
from PIL import Image
import numpy as np
import swanlab
from einops import rearrange, reduce, repeat

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm, BatchNorm1d
from torch.nn import MultiheadAttention

from torchvision import transforms
from torchsummary import summary

from modelscope import ViTImageProcessor, ViTModel, BartTokenizer, BartModel

import transformers
import warnings
from readData import MyDataset

warnings.filterwarnings('ignore')   

def bart_extractor(tokenizer, t_model, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    t_outputs = t_model(**inputs)
    t_last_hidden_states = t_outputs.last_hidden_state
    # ViT 和 BART 的输出中，第一个 token（CLS token）通常包含整个序列的语义信息。通过只取 CLS token 的特征，可以将序列特征压缩为一个固定大小的向量。
    return t_last_hidden_states[:, 0, :]

def vit_extractor(processor, v_model, image):
    inputs = processor(images=image, return_tensors="pt")
    v_outputs = v_model(**inputs)
    v_last_hidden_states = v_outputs.last_hidden_state
    return v_last_hidden_states[:, 0, :]

class Cross_modal_atten(nn.Module): 
    def __init__(self, d_model=64, nhead=8, dropout=0.1,
                 layer_norm_eps=1e-5, First = False,
                 device=None, dtype=None) -> None:

        super(Cross_modal_atten, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        if First == True:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) 
        self.norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)  
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True,
                                            **factory_kwargs)
        self.dropout = Dropout(dropout) 
        self.First = First

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        # print("Cross_modal_atten x1 shape:", x1.shape)
        # print("Cross_modal_atten x2 shape:", x2.shape)
        
        x = torch.cat((x1, x2), dim=1)
        # print("Concatenated x shape:", x.shape)
        b = x.shape[0]
        if self.First == True:
            cls_tokens = repeat(self.cls_token, '() s e -> b s e', b=b)
            # prepend the cls token to the input
            # print("cls_tokens shape:", cls_tokens.shape)
            # print("x shape:", x.shape)
            x = torch.reshape(x, (b, -1, x.shape[-1]))  # Ensure x is 3D
            # print("Reshaped x shape:", x.shape)
            src = torch.cat([cls_tokens, x], dim=1)
        else:
            src = x
        src2 = self.cross_attn(src, src, src)[0]
        out = src + self.dropout(src2)
        out = self.norm(out)
        # print("Cross_modal_atten Output shape:", out.shape)
        out = out[:,0,:].unsqueeze(dim=1)

        return out

class Feed_forward(nn.Module): 
    def __init__(self, d_model=64, dropout=0.1, dim_feedforward=512,
                 num_classes=2, layer_norm_eps=1e-5,
                 device=None, dtype=None) -> None:

        super(Feed_forward, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.relu = nn.ReLU()
        self.dropout1 = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, dim_feedforward, **factory_kwargs)
        self.dropout2 = Dropout(dropout)
        self.linear3 = Linear(dim_feedforward, num_classes, **factory_kwargs)
        # self.norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        
    def forward(self, x: Tensor) -> Tensor:   
        # print("Feed_forward Input x shape:", x.shape)
        out = self.dropout1(self.relu(self.linear1(x)))
        out = self.dropout2(self.relu(self.linear2(out)))
        out = self.linear3(out)
        # out = self.norm(out)
        # print("Feed_forward Output shape:", out.shape)
        out = out.squeeze(dim=1)  # Remove the sequence dimension
        # print("Feed_forward Output after squeeze shape:", out.shape)
        return out


class Cross_Transformer(nn.Module):
    def __init__(self, d_model=64, nhead=8, dropout=0.1,
                 dim_feedforward=512, num_classes=2,
                 layer_norm_eps=1e-5, device=None, dtype=None):
        super(Cross_Transformer, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.cross_modal_atten = Cross_modal_atten(d_model=d_model, nhead=nhead, dropout=dropout,
                                                   layer_norm_eps=layer_norm_eps, First=True, **factory_kwargs)
        self.feed_forward = Feed_forward(d_model=d_model, dropout=dropout, dim_feedforward=dim_feedforward,
                                         num_classes=num_classes, layer_norm_eps=layer_norm_eps, **factory_kwargs)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = self.cross_modal_atten(x1, x2)
        out = self.feed_forward(x)
        return out


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 80 epochs"""
    lr = init_lr * (0.1 ** (epoch // 80))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    

    '''
    parmeters:
    '''
    batchsz = 512
    epochs = 100
    init_lr = 0.001
    input_size = 1536  # BERT and ViT output size

    run = swanlab.init(
        project="depression_detection",
        experiment_name="20250619",
        description="Cross-Modal Transformer for Depression Detection",
        )
    
    swanlab.config={"epochs": epochs, "learning_rate": init_lr, "batch_size": batchsz, "model_type": "multimodal_transformer",
                   "input_size": input_size, "d_model": input_size, "nhead": 8, "dropout": 0.1, "dim_feedforward": 512, "num_classes": 2}
    train_data_dir = "/nfs/xy_outputs/depression/"
    data_transform = {
    "train": transforms.Compose([transforms.Resize([512, 512]),
                                 transforms.RandomRotation(10),
                                 transforms.RandomResizedCrop(512, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
                                 transforms.ToTensor(),]),
    "val": transforms.Compose([transforms.Resize([512, 512]),
                               transforms.ToTensor(),])
                    }
    
    train_datasets = MyDataset(train_data_dir + "train.txt", transform=data_transform["train"])
    traindataloader = DataLoader(train_datasets, batch_size=batchsz, shuffle=True, num_workers=0)
    test_datasets = MyDataset(train_data_dir + "test.txt", transform=data_transform["val"])
    test_num = len(test_datasets)
    testdataloader = DataLoader(test_datasets, batch_size=batchsz, shuffle=False, num_workers=0)
    
    
    # model = Feed_forward(d_model=input_size, dropout=0.1, dim_feedforward=512)
    model = Cross_Transformer(d_model=input_size, nhead=8, dropout=0.1,
                              dim_feedforward=512, num_classes=2, layer_norm_eps=1e-5)
    model = model.to(device)
    print(model)
    summary(model, [(1, input_size),(1, input_size)], device=device)

    vit_model = "google/vit-base-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(vit_model)
    v_model = ViTModel.from_pretrained(vit_model)
    bart_model = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(bart_model)
    t_model = BartModel.from_pretrained(bart_model)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    best_acc = 0.0
    save_path = './weights/best.pth'
    save_path_every = './weights/last.pth'
    for epoch in range(epochs):
        print(f"*********************************> Epoch {epoch+1}/{epochs} <*********************************")
        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch)
        model.train()
        running_loss = 0.0
        for i, (image, text, label) in enumerate(traindataloader):
            print("-------> Train batch id:", i)
            
            image = image.to(device)
            # print("Image shape:", image.shape)
            text = list(text)
            # print("Text shape:", len(text))
            label = label.to(device)

            v_out = vit_extractor(processor, v_model, image)
            t_out = bart_extractor(tokenizer, t_model, text)
            # print("Image feature shape:", v_out.shape)
            # print("Text feature shape:", t_out.shape)
            # v_out = v_out.squeeze(0)
            # t_out = t_out.squeeze(0)
            # cat_out = torch.cat((v_out, t_out), dim=1)
            # print("Concatenated feature shape:", cat_out.shape)
            # cat_out = cat_out.view(cat_out.size(0), -1)  # Flatten the output if needed
            # print("Flattened feature shape:", cat_out.shape)

            # outputs = model(cat_out.to(device))
            outputs = model(v_out.to(device), t_out.to(device))
            # print("Output shape:", outputs.shape)
            # print("Label shape:", label.shape)
            loss = loss_function(outputs, label)
            print(f"Loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            swanlab.log({ "loss": loss})
            running_loss += loss.item()
            
        swanlab.log({ "running_loss": running_loss})
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(traindataloader):.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), save_path_every.replace('last.pth', f'{epoch+1}.pth'))  # save every 10 epochs
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (image, text, label) in enumerate(testdataloader):
                print(f"+++++++>Test batch id: {i}")
                image = image.to(device)
                text = list(text)
                label = label.to(device)

                v_out = vit_extractor(processor, v_model, image)
                t_out = bart_extractor(tokenizer, t_model, text)
                # cat_out = torch.cat((v_out, t_out), dim=1)
                # outputs = model(cat_out)
                outputs = model(v_out.to(device), t_out.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                print(f"Correct num is:{correct}")
        acc = 100 * correct / total
        swanlab.log({"test_accuracy": acc})
        print(f"Test Accuracy: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with accuracy: {best_acc:.2f}%")
    torch.save(model.state_dict(), save_path_every)  # save the last model
    print(f"Training complete. Best accuracy: {best_acc:.2f}%")
    swanlab.finish()

    
    


    
    
