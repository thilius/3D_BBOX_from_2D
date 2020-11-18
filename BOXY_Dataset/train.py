import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import argparse


from model import Net
from utils.dataset import ReadDataset


def train(train_dataloader, model, mse_loss, optimizer, epoch, writer, config):
    model.train()
    running_loss = 0.
    total_average_loss = 0.

    for batch_i, (_, imgs, targets) in enumerate(train_dataloader):
        imgs = Variable(imgs.float()).cuda()
        targets = Variable(targets.view((-1, 8)).float(), requires_grad=False).cuda()

        optimizer.zero_grad()

        output = model(imgs)


        loss = mse_loss(output, targets)

        loss.backward()
        optimizer.step()

        print('[Epoch %d/%d, Batch %d/%d] [Loss: total %f]'
              %
              (epoch,
              config.epochs,
              batch_i,
              len(train_dataloader),
              loss.item())
              )

        #save the loss every 100 iterations

        running_loss += loss.item()
        total_average_loss += loss.item()

        if batch_i % 100 == 99:
            global_step = batch_i + epoch * len(train_dataloader) + 1
            writer.add_scalar('Train Loss', running_loss / 100, global_step)
            writer.flush()
            running_loss = 0.



    #save the weights of every epoch
    if epoch % config.checkpoint_interval == 0:
        total_average_loss /= len(train_dataloader)
        torch.save(model.state_dict(), '%s/epoch_%d_train_%.6f.pth' % (config.checkpoint_dir, epoch + 1, total_average_loss))


def valid(val_dataloader, model, mse_loss, epoch, writer, config):
    print('now eval on val dataset.................................................')
    model.eval()
    running_loss = 0.

    with torch.no_grad():
        for batch_i, (_, imgs, targets) in enumerate(val_dataloader):
            imgs = Variable(imgs).cuda()
            targets = Variable(targets.view(-1, 8), requires_grad=False).cuda()

            output = model(imgs)

            loss = mse_loss(output, targets)

            running_loss += loss.item()

        writer.add_scalar('Val Loss', running_loss / len(val_dataloader), epoch + 1)
        writer.flush()

        pth_filenames = os.listdir(config.checkpoint_dir)
        for filename in pth_filenames:
            if 'epoch_%d' % (epoch+1) in filename:
                pth_oldname = config.checkpoint_dir + '/' + filename
                pth_newname = os.path.splitext(pth_oldname)[0] + '_val_%.6f.pth' % (running_loss / len(val_dataloader))
                os.rename(pth_oldname, pth_newname)
        return running_loss/len(val_dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='size of each img batch')
    parser.add_argument('--train_path', type=str, default='train.txt', help='train dataset path')
    parser.add_argument('--val_path', type=str, default='valid.txt', help='valid dataset path')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between saving model weights')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='directory for saving model weights')
    parser.add_argument('--tensorboard_dir', type=str, default='logs', help='directory for saving logs')
    config = parser.parse_args()
    print(config)

    torch.set_num_threads(4)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.tensorboard_dir, exist_ok=True)

    writer = SummaryWriter(config.tensorboard_dir)

    model = Net().cuda()

    train_dataloader = DataLoader(ReadDataset(config.train_path),
                                  batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    val_dataloader = DataLoader(ReadDataset(config.val_path),
                                batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    mse_loss = torch.nn.MSELoss().cuda()

    min_valloss = 100000
    early_stop = 0
    for epoch in range(config.epochs):
        train(train_dataloader, model, mse_loss, optimizer, epoch, writer, config)

        if config.val_path:
            valloss = valid(val_dataloader, model, mse_loss, epoch, writer, config)
            if valloss > min_valloss:
                early_stop += 1
                if early_stop == 5:
                    break
            else:
                min_valloss = valloss
        scheduler.step()

    writer.close()


if __name__ == '__main__':
    main()
