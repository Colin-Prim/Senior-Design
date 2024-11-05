import os
import torch
import torch.utils.data
from joints_detectors.Alphapose.train_sppe.src.utils.dataset import coco
from joints_detectors.Alphapose.train_sppe.src.opt import opt  # Ensure this path is correct
from tqdm import tqdm
from .models.FastPose import createModel
from joints_detectors.Alphapose.train_sppe.src.utils.eval import DataLogger, accuracy
from joints_detectors.Alphapose.train_sppe.src.utils.img import flip, shuffleLR
from joints_detectors.Alphapose.train_sppe.src.evaluation import prediction

from tensorboardX import SummaryWriter
import os


def check_h5_file():
    h5_file_path = './joint_detectors/Alphapose/train_sppe/data/coco/annot_coco.h5'
    if not os.path.exists(h5_file_path):
        print(f"The {h5_file_path} file is missing. Please check the file path.")
        exit(1)
    else:
        print(f"The {h5_file_path} file is found.")


def train(train_loader, m, criterion, optimizer, writer):
    lossLogger = DataLogger()
    accLogger = DataLogger()
    m.train()

    train_loader_desc = tqdm(train_loader)

    for i, (inps, labels, setMask, imgset) in enumerate(train_loader_desc):
        inps = inps.cuda().requires_grad_()
        labels = labels.cuda()
        setMask = setMask.cuda()
        out = m(inps)

        loss = criterion(out.mul(setMask), labels)

        acc = accuracy(out.data.mul(setMask), labels.data, train_loader.dataset)

        accLogger.update(acc[0], inps.size(0))
        lossLogger.update(loss.item(), inps.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        opt.trainIters += 1
        # Tensorboard
        writer.add_scalar(
            'Train/Loss', lossLogger.avg, opt.trainIters)
        writer.add_scalar(
            'Train/Acc', accLogger.avg, opt.trainIters)

        # TQDM
        train_loader_desc.set_description(
            'loss: {loss:.8f} | acc: {acc:.2f}'.format(
                loss=lossLogger.avg,
                acc=accLogger.avg * 100)
        )

    train_loader_desc.close()

    return lossLogger.avg, accLogger.avg


def valid(val_loader, m, criterion, optimizer, writer):
    lossLogger = DataLogger()
    accLogger = DataLogger()
    m.eval()

    val_loader_desc = tqdm(val_loader)

    for i, (inps, labels, setMask, imgset) in enumerate(val_loader_desc):
        inps = inps.cuda()
        labels = labels.cuda()
        setMask = setMask.cuda()

        with torch.no_grad():
            out = m(inps)

            loss = criterion(out.mul(setMask), labels)

            flip_out = m(flip(inps))
            flip_out = flip(shuffleLR(flip_out, val_loader.dataset))

            out = (flip_out + out) / 2

        acc = accuracy(out.mul(setMask), labels, val_loader.dataset)

        lossLogger.update(loss.item(), inps.size(0))
        accLogger.update(acc[0], inps.size(0))

        opt.valIters += 1

        # Tensorboard
        writer.add_scalar(
            'Valid/Loss', lossLogger.avg, opt.valIters)
        writer.add_scalar(
            'Valid/Acc', accLogger.avg, opt.valIters)

        val_loader_desc.set_description(
            'loss: {loss:.8f} | acc: {acc:.2f}'.format(
                loss=lossLogger.avg,
                acc=accLogger.avg * 100)
        )

    val_loader_desc.close()

    return lossLogger.avg, accLogger.avg


def main():
    # Check if the annot_coco.h5 file exists
    check_h5_file()

    # Model Initialize
    m = createModel().cuda()
    if opt.loadModel:
        print('Loading Model from {}'.format(opt.loadModel))
        m.load_state_dict(torch.load(opt.loadModel))
        if not os.path.exists("../exp/{}/{}".format(opt.dataset, opt.expID)):
            try:
                os.mkdir("../exp/{}/{}".format(opt.dataset, opt.expID))
            except FileNotFoundError:
                os.mkdir("../exp/{}".format(opt.dataset))
                os.mkdir("../exp/{}/{}".format(opt.dataset, opt.expID))
    else:
        print('Create new model')
        if not os.path.exists("../exp/{}/{}".format(opt.dataset, opt.expID)):
            try:
                os.mkdir("../exp/{}/{}".format(opt.dataset, opt.expID))
            except FileNotFoundError:
                os.mkdir("../exp/{}".format(opt.dataset))
                os.mkdir("../exp/{}/{}".format(opt.dataset, opt.expID))

    criterion = torch.nn.MSELoss().cuda()

    if opt.optMethod == 'rmsprop':
        optimizer = torch.optim.RMSprop(m.parameters(),
                                        lr=opt.LR,
                                        momentum=opt.momentum,
                                        weight_decay=opt.weightDecay)
    elif opt.optMethod == 'adam':
        optimizer = torch.optim.Adam(
            m.parameters(),
            lr=opt.LR
        )
    else:
        raise Exception

    writer = SummaryWriter(
        '.tensorboard/{}/{}'.format(opt.dataset, opt.expID))

    # Prepare Dataset
    if opt.dataset == 'coco':
        train_dataset = coco.Mscoco(train=True)
        val_dataset = coco.Mscoco(train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.trainBatch, shuffle=True, num_workers=opt.nThreads, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.validBatch, shuffle=False, num_workers=opt.nThreads, pin_memory=True)

    # Model Transfer
    m = torch.nn.DataParallel(m).cuda()

    # Start Training
    for i in range(opt.nEpochs):
        opt.epoch = i

        print('############# Starting Epoch {} #############'.format(opt.epoch))
        loss, acc = train(train_loader, m, criterion, optimizer, writer)

        print('Train-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
            idx=opt.epoch,
            loss=loss,
            acc=acc
        ))

        opt.acc = acc
        opt.loss = loss
        m_dev = m.module
        if i % opt.snapshot == 0:
            torch.save(
                m_dev.state_dict(), '../exp/{}/{}/model_{}.pkl'.format(opt.dataset, opt.expID, opt.epoch))
            torch.save(
                opt, '../exp/{}/{}/option.pkl'.format(opt.dataset, opt.expID, opt.epoch))
            torch.save(
                optimizer, '../exp/{}/{}/optimizer.pkl'.format(opt.dataset, opt.expID))

        loss, acc = valid(val_loader, m, criterion, optimizer, writer)

        print('Valid-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
            idx=i,
            loss=loss,
            acc=acc
        ))

    writer.close()


if __name__ == '__main__':
    main()
