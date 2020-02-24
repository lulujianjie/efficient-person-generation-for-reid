import logging
import time
import torch
import torch.nn as nn
import cv2
from utils.meter import AverageMeter
from utils.image_pool import ImagePool
from utils.metrics import ssim_score
import numpy as np


def loss_reid_factor(epoch):
    if epoch <= 50:
        factor = 0
    elif epoch > 50 and epoch <= 150:
        factor = 1.0/100*epoch - 0.5
    else:
        factor = 1.0
    return factor

def do_train(
        Cfg,
        model_G,model_Dip,model_Dii,model_D_reid,
        train_loader,val_loader,
        optimizerG,optimizerDip,optimizerDii,
        GAN_loss, L1_loss, ReID_loss,
        schedulerG, schedulerDip, schedulerDii
        ):
    log_period = Cfg.SOLVER.LOG_PERIOD
    checkpoint_period = Cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = Cfg.SOLVER.EVAL_PERIOD
    output_dir = Cfg.DATALOADER.LOG_DIR
    # need modified the following in cfg
    epsilon = 0.00001
    margin = 0.4
    ####################################
    device = "cuda"
    epochs = Cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger('pose-transfer-gan.train')
    logger.info('Start training')

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model_G = nn.DataParallel(model_G)
            model_Dii = nn.DataParallel(model_Dii)
            model_Dip = nn.DataParallel(model_Dip)
        model_G.to(device)
        model_Dip.to(device)
        model_Dii.to(device)
        model_D_reid.to(device)
    lossG_meter = AverageMeter()
    lossDip_meter = AverageMeter()
    lossDii_meter = AverageMeter()
    distDreid_meter = AverageMeter()
    fake_ii_pool = ImagePool(50)
    fake_ip_pool = ImagePool(50)

    #evaluator = R1_mAP(num_query, max_rank=50, feat_norm=Cfg.TEST.FEAT_NORM)
    #train
    for epoch in range(1, epochs+1):
        start_time = time.time()
        lossG_meter.reset()
        lossDip_meter.reset()
        lossDii_meter.reset()
        distDreid_meter.reset()
        schedulerG.step()
        schedulerDip.step()
        schedulerDii.step()

        model_G.train()
        model_Dip.train()
        model_Dii.train()
        model_D_reid.eval()
        for iter, batch in enumerate(train_loader):
            img1 = batch['img1'].to(device)
            pose1 = batch['pose1'].to(device)
            img2 = batch['img2'].to(device)
            pose2 = batch['pose2'].to(device)
            input_G = (img1,pose2)

            #forward
            fake_img2 = model_G(input_G)
            optimizerG.zero_grad()

            #train G
            input_Dip = torch.cat((fake_img2, pose2), 1)
            pred_fake_ip = model_Dip(input_Dip)
            loss_G_ip = GAN_loss(pred_fake_ip, True)
            input_Dii = torch.cat((fake_img2, img1), 1)
            pred_fake_ii = model_Dii(input_Dii)
            loss_G_ii = GAN_loss(pred_fake_ii, True)

            loss_L1,_,_ = L1_loss(fake_img2, img2)

            feats_real = model_D_reid(img2)
            feats_fake = model_D_reid(fake_img2)

            dist_cos = torch.acos(torch.clamp(torch.sum(feats_real * feats_fake, 1), -1+ epsilon, 1- epsilon))

            same_id_tensor = torch.FloatTensor(dist_cos.size()).fill_(1).to('cuda')
            dist_cos_margin = torch.max(dist_cos - margin, torch.zeros_like(dist_cos))
            loss_reid = ReID_loss(dist_cos_margin, same_id_tensor)
            factor = loss_reid_factor(epoch)
            loss_G = 0.5*loss_G_ii*Cfg.LOSS.GAN_WEIGHT + 0.5*loss_G_ip*Cfg.LOSS.GAN_WEIGHT+loss_L1 + loss_reid*Cfg.LOSS.REID_WEIGHT*factor
            loss_G.backward()
            optimizerG.step()

            #train Dip
            for i in range(Cfg.SOLVER.DG_RATIO):
                optimizerDip.zero_grad()
                real_input_ip = torch.cat((img2, pose2), 1)
                fake_input_ip = fake_ip_pool.query(torch.cat((fake_img2, pose2), 1).data)
                pred_real_ip = model_Dip(real_input_ip)
                loss_Dip_real = GAN_loss(pred_real_ip, True)
                pred_fake_ip = model_Dip(fake_input_ip)
                loss_Dip_fake = GAN_loss(pred_fake_ip, False)
                loss_Dip = 0.5*Cfg.LOSS.GAN_WEIGHT*(loss_Dip_real+loss_Dip_fake)
                loss_Dip.backward()
                optimizerDip.step()
            #train Dii
            for i in range(Cfg.SOLVER.DG_RATIO):
                optimizerDii.zero_grad()
                real_input_ii = torch.cat((img2, img1), 1)
                fake_input_ii = fake_ii_pool.query(torch.cat((fake_img2, img1), 1).data)
                pred_real_ii = model_Dii(real_input_ii)
                loss_Dii_real = GAN_loss(pred_real_ii, True)
                pred_fake_ii = model_Dii(fake_input_ii)
                loss_Dii_fake = GAN_loss(pred_fake_ii, False)
                loss_Dii = 0.5*Cfg.LOSS.GAN_WEIGHT*(loss_Dii_real+loss_Dii_fake)
                loss_Dii.backward()
                optimizerDii.step()

            lossG_meter.update(loss_G.item(), 1)
            lossDip_meter.update(loss_Dip.item(), 1)
            lossDii_meter.update(loss_Dii.item(), 1)
            distDreid_meter.update(dist_cos.mean().item(), 1)
            if (iter+1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] G Loss: {:.3f}, Dip Loss: {:.3f}, Dii Loss: {:.3f}, Base G_Lr: {:.2e}, Base Dip_Lr: {:.2e}, Base Dii_Lr: {:.2e}"
                            .format(epoch, (iter+1), len(train_loader),lossG_meter.avg, lossDip_meter.avg, lossDii_meter.avg,
                                    schedulerG.get_lr()[0], schedulerDip.get_lr()[0], schedulerDii.get_lr()[0]))#scheduler.get_lr()[0]
                logger.info("ReID Cos Distance: {:.3f}".format(distDreid_meter.avg))
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            torch.save(model_G.state_dict(), output_dir+'model_G_{}.pth'.format(epoch))
            torch.save(model_Dip.state_dict(), output_dir+'model_Dip_{}.pth'.format(epoch))
            torch.save(model_Dii.state_dict(), output_dir+'model_Dii_{}.pth'.format(epoch))
        #
        if epoch % eval_period == 0:
            np.save(output_dir+'train_Bx6x128x64_epoch{}.npy'.format(epoch), fake_ii_pool.images[0].cpu().numpy())
            logger.info('Entering Evaluation...')
            tmp_results = []
            model_G.eval()
            for iter, batch in enumerate(val_loader):
                with torch.no_grad():
                    img1 = batch['img1'].to(device)
                    pose1 = batch['pose1'].to(device)
                    img2 = batch['img2'].to(device)
                    pose2 = batch['pose2'].to(device)
                    input_G = (img1, pose2)
                    fake_img2 = model_G(input_G)
                    tmp_result = torch.cat((img1, img2, fake_img2), 1).cpu().numpy()
                    tmp_results.append(tmp_result)

            np.save(output_dir + 'test_Bx6x128x64_epoch{}.npy'.format(epoch), tmp_results[0])

def do_inference(Cfg,
                 model_G,
                 val_loader):
    output_dir = Cfg.DATALOADER.LOG_DIR
    device = "cuda"
    logger = logging.getLogger("pose-transfer-gan.test")

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model_G = nn.DataParallel(model_G)
        model_G.to(device)

    logger.info('Entering Evaluation...')
    tmp_results = []
    img_path1_paths = []
    img_path2_paths = []
    model_G.eval()
    for iter, batch in enumerate(val_loader):
        with torch.no_grad():
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)
            pose2 = batch['pose2'].to(device)
            img_path1 = batch['img_path1']
            img_path2 = batch['img_path2']
            input_G = (img1, pose2)
            fake_img2 = model_G(input_G)
            #print(fake_img2.shape)
            tmp_result = torch.cat((img2, fake_img2), 1).cpu().numpy()
            tmp_results.append(tmp_result)
            img_path1_paths.append(img_path1)
            img_path2_paths.append(img_path2)
    logger.info('Finished Evaluation...')
    np.save(output_dir + 'result_Bx6x128x64.npy', tmp_results)

    target_images = []
    generated_images = []
    for i, batch in enumerate(tmp_results):
        for idx in range(batch.shape[0]):
            img2 = (np.transpose(batch[idx, 0:3, :, :], (1, 2, 0)) + 1) / 2.0 * 255.0
            fake_img2 = (np.transpose(batch[idx, 3:6, :, :], (1, 2, 0)) + 1) / 2.0 * 255.0
            target_images.append(img2.astype(np.int))
            generated_images.append(fake_img2.astype(np.int))
            print(img_path1_paths[i][idx],img_path2_paths[i][idx])
            cv2.imwrite(Cfg.TEST.GT_PATH+'{}_{}.png'.format(img_path1_paths[i][idx],img_path2_paths[i][idx]), cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
            cv2.imwrite(Cfg.TEST.GENERATED_PATH+'{}_{}.png'.format(img_path1_paths[i][idx],img_path2_paths[i][idx]), cv2.cvtColor(fake_img2, cv2.COLOR_RGB2BGR))

    logger.info("Compute structured similarity score (SSIM)...")
    structured_score = ssim_score(generated_images, target_images)
    logger.info("SSIM score %s" % structured_score)