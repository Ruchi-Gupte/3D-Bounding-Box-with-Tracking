from utils.Dataset import Dataset
from tqdm import tqdm
import os
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from Model import Model


def plot_graph(train_loss_curve):
    # Training Loss vs Epochs
    plt.plot(range(len(train_loss_curve)), train_loss_curve, label="Train Loss")
    plt.title("Epoch Vs Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("Loss_curves.jpg")
    plt.show()

def OrientationLoss(orient_batch, orientGT_batch, confGT_batch):
    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]

    # extract just the important bin
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    orient_batch = orient_batch[torch.arange(batch_size), indexes]

    theta_diff = torch.atan2(orientGT_batch[:,1], orientGT_batch[:,0])
    estimated_theta_diff = torch.atan2(orient_batch[:,1], orient_batch[:,0])

    return -1 * torch.cos(theta_diff - estimated_theta_diff).mean()

def calc_loss(orient, conf, dim, labels, device):
    gt_orient = labels['Orientation'].float().to(device)
    gt_conf = labels['Confidence'].long().to(device)
    gt_dim = labels['Dimensions'].float().to(device)

    alpha = 0.6
    w = 0.4

    orient_loss = OrientationLoss(orient, gt_orient, gt_conf)
    dim_loss = torch.nn.functional.mse_loss(dim, gt_dim)

    gt_conf = torch.max(gt_conf, dim=1)[1]
    conf_loss = torch.nn.functional.cross_entropy(conf, gt_conf)

    loss_theta = conf_loss + w * orient_loss
    return alpha * dim_loss + loss_theta


def train_test_model(epochs, device, use_saved=False):
    root = os.path.abspath(os.path.dirname(__file__))
    train_path = root + '/Kitti/training'
    save_dir = root + '/trained_models/'
    model_path = root + '/trained_models/model_epoch_0.pth'
    dataset = Dataset(train_path)
    print("Obtained Training Data")

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=6)

    backbone = torchvision.models.vgg.vgg19_bn(pretrained=True)
    model = Model(features=backbone.features).to(device)
    opt_SGD = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    start = 0

    if use_saved:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start = checkpoint['epoch']

        print('Found previous checkpoint at epoch ', start)
        print('Resuming training....') 

    train_loss_curve = []
    for epoch in range(start, start+epochs):
        training_loss = 0
        for i, (data, labels) in enumerate(tqdm(trainloader)):
            data=data.float().to(device)
            orient, conf, dim = model(data)
            loss = calc_loss(orient, conf, dim, labels, device)
            training_loss += loss.item()
            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()
            

        train_loss_curve.append(training_loss/len(trainloader))
        print("Epoch: ",epoch+1," Training Loss:", training_loss/len(trainloader))

        # Save Model after each epoch
        name = save_dir + 'model_epoch_%s.pth' % epoch
        torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict()}, name)
    return train_loss_curve

if __name__=='__main__':
    epochs =2
    use_saved = True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        cudnn.benchmark = True
    train_loss_curve= train_test_model(epochs, device, use_saved)
    plot_graph(train_loss_curve)
