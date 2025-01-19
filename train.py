import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from vit_model_2018 import vit_ as create_model  # 自己的方法
from my_dataset import MyGLUDataset
import pandas as pd
from torch.optim import lr_scheduler
from torchsummary import summary
import time
# 设置随机种子
torch.manual_seed(1)

# 如果你在使用CUDA，还需要设置这个
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
# 设置随机种子
np.random.seed(1)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
save_path = "./model"
classes = ['32PSK',
 '16APSK',
 '32QAM',
 'FM',
 'GMSK',
 '32APSK',
 'OQPSK',
 '8ASK',
 'BPSK',
 '8PSK',
 'AM-SSB-SC',
 '4ASK',
 '16PSK',
 '64APSK',
 '128QAM',
 '128APSK',
 'AM-DSB-SC',
 'AM-SSB-WC',
 '64QAM',
 'QPSK',
 '256QAM',
 'AM-DSB-WC',
 'OOK',
 '16QAM']

# 创建train_acc.csv和var_acc.csv文件，记录loss和accuracy
df = pd.DataFrame(columns=['epoch', 'train Loss', 'training accuracy', 'test accuracy'])  # 列名
df.to_csv("./log/vit_acc/vit_2018_patch512_n4_head4_train_acc_12_18.csv", index=False)  # 路径可以根据需要更改

train_data_set = MyGLUDataset(mode="train")
val_data_set = MyGLUDataset(mode="test")

# if __name__ == 'main':

batch_size = 32
# nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
# print('Using {} dataloader workers'.format(nw))
train_loader = torch.utils.data.DataLoader(train_data_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           # num_workers=nw,
                                           collate_fn=train_data_set.collate_fn)

val_loader = torch.utils.data.DataLoader(val_data_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           # num_workers=nw,
                                           collate_fn=val_data_set.collate_fn)
val_num = len(val_data_set)
train_num = len(train_data_set)
train_steps = len(train_loader)
model = create_model(24)  # vit2019

print(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# 定义优化器和损失函数
criterion = nn.CrossEntropyLoss()  # 用交叉熵会出错

optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.005)# vit2018使用


model.to(device)
summary(model, input_size=(1, 2, 1024))
print("没有问题")
epochs = 100
best_acc = 0
# 余弦退火
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-4, last_epoch=-1)
# if __name__ == 'main':
for epoch in range(epochs):
    # 统计时间
    since = time.time()
    # train
    model.train()
    running_loss = 0.0
    train_acc = 0
    val_loss = 0
    for step, data in enumerate(train_loader):
        images, labels = data  # [8, 2, 1, 1024] [8, 24]
        labels = labels.type(torch.FloatTensor)

        optimizer.zero_grad()
        logits = model(images.to(device))  # [8, 24]
        predict_y = torch.max(logits, dim=1)[1]
        label_y = torch.max(labels, dim=1)[1]
        train_acc += torch.eq(predict_y, label_y.to(device)).sum().item()

        loss = criterion(logits, labels.to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()  # 余弦退火

        # print statistics
        running_loss += loss.item()

    print("Finish train")
    # 打印每个epoch的用时
    time_elapsed = time.time() - since
    print("second/epoch:", time_elapsed)
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('[epoch %d] train_loss: %.3f  train_acc: %.3f' %
          (epoch + 1, running_loss / train_steps, train_acc/train_num))
    # validate
    model.eval()
    acc = 0.0  # accumulate accurate number / epoch
    # acc_adv = 0
    loss_count = 0
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data
            val_labels = val_labels.type(torch.FloatTensor)
            outputs = model(val_images.to(device))
            loss_count += criterion(outputs, val_labels.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            label_y = torch.max(val_labels, dim=1)[1]
            torch.eq(predict_y, label_y.to(device))
            acc += torch.eq(predict_y, label_y.to(device)).sum().item()

    val_loss = loss_count / val_num
    val_accurate = acc / val_num

    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f  val_loss: %.3f' %
          (epoch + 1, running_loss / train_steps, val_accurate, val_loss))

    epochs_ji = "Epoch[%d]" % (epoch+1)
    train_loss_ji = "%f" % (running_loss / train_steps)
    train_acc_ji = "%g" % (train_acc/train_num)
    test_acc_ji = "%g" % val_accurate
    # 将数据保存在一维列表
    list = [epochs_ji, train_loss_ji, train_acc_ji, test_acc_ji]
    # 由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
    data = pd.DataFrame([list])
    data.to_csv("./log/vit_acc/vit_2018_patch512_n4_head4_train_acc_12_18.csv", mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
    if val_accurate > best_acc:
        best_acc = val_accurate
        #nishiyongba
        # torch.save(model.state_dict(), save_path)
torch.save(model.state_dict(), "./model/vit2018/base_all" + "/" + str(best_acc) + ".pt")
print("Best validation accuracy:", best_acc)
filename = "./log/vit_acc/vit_2018_patch512_n4_head4_train_acc_12_18.csv"
df = pd.read_csv(filename, index_col="epoch")
ax = df.plot()
fig = ax.get_figure()
fig.savefig('./log/vit_acc/vit_2018_patch512_n4_head4_train_acc_12_18.jpg')





