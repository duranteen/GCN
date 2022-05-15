## 模型训练
import numpy as np
import torch.cuda
import torch
from torch import nn
from CoraData import CoraData
from model.GCN import GCN

learning_rate = 0.1
weight_decay = 5e-4
num_epochs = 200
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CoraData().data
node_feature = dataset.x / dataset.x.sum(1, keepdims=True) #归一化数据
tensor_x = torch.from_numpy(node_feature).to(device)
tensor_y = torch.from_numpy(dataset.y).to(device)
tensor_train_mask = torch.from_numpy(dataset.train_mask).to(device)
tensor_val_mask = torch.from_numpy(dataset.val_mask).to(device)
tenso_test_mask = torch.from_numpy(dataset.test_mask).to(device)
normalized_adj = CoraData.normalization(dataset.adjacency)

num_nodes, input_dim = node_feature.shape
indices = torch.from_numpy(np.asarray([normalized_adj.row,
                                       normalized_adj.col]).astype("int64")).long()
values = torch.from_numpy(normalized_adj.data.astype(np.float32))
tensor_adjacency = torch.sparse.FloatTensor(indices, values, (num_nodes, num_nodes)).to(device)

model = GCN(input_dim).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def test(mask):
    model.eval()
    with torch.no_grad():
        logits = model(tensor_adjacency, tensor_x)
        test_mask_logits = logits[mask]
        predict_y = test_mask_logits.max(1)[1]
        accuracy = torch.eq(predict_y, tensor_y[mask]).float().mean()
    return accuracy, test_mask_logits.cpu().numpy(), tensor_y[mask].cpu().numpy()


def train():
    loss_history = []
    val_acc_history = []
    model.train()
    train_y = tensor_y[tensor_train_mask]
    for epoch in range(num_epochs):
        logits = model(tensor_adjacency, tensor_x)
        train_mask_logits = logits[tensor_train_mask]
        loss = criterion(train_mask_logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc, _, _ = test(tensor_train_mask)
        val_acc, _, _ = test(tensor_val_mask)
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4f}, ValAcc {:.4f}".format(
            epoch, loss.item(), train_acc.item(), val_acc.item()
        ))
    return loss_history, val_acc_history


from matplotlib import pyplot as plt
def plot_loss_and_acc(loss_history, acc_history):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)
    plt.ylabel('Loss')

    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(acc_history)), acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()


loss, val_acc = train()
test_acc, test_logits, test_label = test(tenso_test_mask)
print("Test accuracy: ", test_acc.item())

plot_loss_and_acc(loss, val_acc)

from sklearn.manifold import TSNE
tsne = TSNE()
out = tsne.fit_transform(test_logits)
fig = plt.figure()
for i in range(7):
    indices = test_label == i
    x, y = out[indices].T
    plt.scatter(x, y, label=str(i))
plt.legend()
plt.show()