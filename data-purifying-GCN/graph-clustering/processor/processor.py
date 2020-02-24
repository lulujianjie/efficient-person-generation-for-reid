import logging
import time
import torch
import torch.nn as nn

from utils.meter import AverageMeter
from utils.metrics import accuracy
import numpy as np
import logging

from utils.metrics import Dist_Mat

def make_labels(gtmat):
    return gtmat.view(-1)


def do_train(Cfg, model, train_loader, test_loader, optimizer, scheduler, loss_fn):
    log_period = Cfg.LOG_PERIOD
    checkpoint_period = Cfg.CHECKPOINT_PERIOD
    output_dir = Cfg.LOG_DIR

    device = "cuda"
    epochs = Cfg.MAX_EPOCHS

    logger = logging.getLogger('{}'.format(Cfg.PROJECT_NAME))
    logger.info('start training')

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()

    #train
    for epoch in range(1, epochs+1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        precision_meter.reset()
        recall_meter.reset()
        scheduler.step()

        model.train()
        for iter, ((feat, adj, cid, h1id), gtmat) in enumerate(train_loader):
            optimizer.zero_grad()
            feat, adj, cid, h1id, gtmat = map(lambda x: x.cuda(),
                                              (feat, adj, cid, h1id, gtmat))
            pred = model(feat, adj, h1id)
            labels = make_labels(gtmat).long()
            loss = loss_fn(pred, labels)
            p, r, acc = accuracy(pred, labels)

            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), feat.size(0))
            acc_meter.update(acc.item(), feat.size(0))
            precision_meter.update(p, feat.size(0))
            recall_meter.update(r, feat.size(0))

            if (iter+1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, P:{:.3f}, R:{:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (iter+1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, precision_meter.avg, recall_meter.avg, scheduler.get_lr()[0]))
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(), output_dir+Cfg.MODEL_NAME+'_{}.pth'.format(epoch))
            model.eval()
            acc_meter.reset()
            precision_meter.reset()
            recall_meter.reset()
            for iter, ((feat, adj, cid, h1id, unique_nodes_list), gtmat) in enumerate(test_loader):
                feat, adj, cid, h1id, gtmat = map(lambda x: x.cuda(),
                                                  (feat, adj, cid, h1id, gtmat))
                pred = model(feat, adj, h1id)
                labels = make_labels(gtmat).long()
                p, r, acc = accuracy(pred, labels)
                acc_meter.update(acc.item(), feat.size(0))
                precision_meter.update(p, feat.size(0))
                recall_meter.update(r, feat.size(0))

            logger.info("Test Result: Acc: {:.3f}, P:{:.3f}, R:{:.3f}"
                        .format(acc_meter.avg, precision_meter.avg, recall_meter.avg))

def do_inference(Cfg, model, test_loader):
    edges = list()
    scores = list()
    device = "cuda"
    logger = logging.getLogger("{}.test".format(Cfg.PROJECT_NAME))
    logger.info("Enter inferencing")

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
    preds = []
    cids = []
    h1ids = []
    model.eval()
    for iter, ((feat, adj, cid, h1id, node_list), gtmat) in enumerate(test_loader):
        feat, adj, cid, h1id, gtmat = map(lambda x: x.cuda(),
                                          (feat, adj, cid, h1id, gtmat))

        pred = model(feat, adj, h1id)

        cids.append(cid.cpu().detach().numpy())
        preds.append(pred.cpu().detach().numpy())
        h1ids.append(h1id.cpu().detach().numpy())

        node_list = node_list.long().squeeze().numpy()
        bs = feat.size(0)
        if bs == 1:
            node_list = np.array([node_list])
        for b in range(bs):
            cidb = cid[b].int().item()
            nl = node_list[b]

            for j,n in enumerate(h1id[b]):
                n = n.item()
                edges.append([nl[cidb], nl[n]])
                scores.append(pred[b*Cfg.NUM_HOP[0]+j,1].item())
        if (iter+1)*Cfg.TEST_BATCHSIZE % 5000 == 0:
            logger.info("Finshed 5000 samples")
    logger.info("Finshed inferencing")
    np.save('./log/preds.npy',preds)
    np.save('./log/cids.npy',cids)
    np.save('./log/h1ids.npy', h1ids)
    # edges = np.asarray(edges)
    # scores = np.asarray(scores)
    # clusters = graph_propagation(edges, scores, max_sz=100, step=0.6, beg_th=0.5, pool='avg')
    # final_pred = clusters2labels(clusters, len(iter+1))
    # np.save('./log/pred_labels.npy', final_pred)
