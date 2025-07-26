# -*- encoding: utf-8 -*-
# @author: yuquanle 
# @time: 2023/04/12
# @version: 0.1
# @description:  Use for civil judgment prediction.

import sys
sys.path.append('/mnt/sdb/leyuquan/github_backup/TML')
import os
import logging
import numpy as np
import pandas as pd
from utils.utils import evaluate_multiclass, set_random_seed, get_precision_recall_f1, labels_to_multihot
from utils.CivilJP_utils import evaluate_civilJP
from models.LawformerCivilJP import KnowAugLawformerCivilJP
from dataset.civilJP_dataset import CivilJPDataset
from configs.KnowAugLawformer_CivilJPConfig import Config
import argparse
import configparser

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AutoTokenizer
import torch.optim as optim
import time
import datetime

from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter  # 导入tensorboard的类
pd.set_option('display.max_columns', None)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    parser = argparse.ArgumentParser(description="KnowAugLawformer_CivilJP")
    # hyper-parameter config.
    parser.add_argument('--dataset_name', type=str, default='cpee', help='[cpee]')
    parser.add_argument('--default_config_path', type=str, default='../configs/KnowAugLawformer_CivilJPConfig.ini', help='../configs/KnowAugLawformer_CivilJPConfig.ini')
    # 必须声明
    parser.add_argument('--local_rank', default=-1, type=int, help='used for distributed parallel')
    parser_args = parser.parse_args()
    
    # Load default config for update each dataset. config_name: model_task_dataset
    args = Config(file_path=parser_args.default_config_path, dataset_name=parser_args.dataset_name, local_rank=parser_args.local_rank,  task='CivilJP')
    # add new parameter
    # args.adapt_param = parser_args.adapt_param

    # print all hyper-paremeter
    logging.info(args)

    torch.cuda.empty_cache()

    # seed random seed
    set_random_seed(args.random_seed)

    # check the device
    # device = 'cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu'
    # logging.info('Using {} device'.format(device))
    # args.device = device'
    
    device = torch.device('cuda')
    # 分布式训练初始化：数据并行
    torch.distributed.init_process_group(backend="nccl")

    logging.info(f'cuda device count: {torch.cuda.device_count()}')
    logging.info(f'world_size: {torch.distributed.get_world_size()}')
    torch.cuda.set_device(args.local_rank)
    args.batch_size = args.batch_size // torch.cuda.device_count()

    # prepare training data
    train_path = '../data/'+ args.dataset_name + '/' + args.dataset_name  +'_train.json'
    valid_path = '../data/'+ args.dataset_name + '/' + args.dataset_name  +'_valid.json'
    test_path = '../data/'+ args.dataset_name + '/' + args.dataset_name  +'_test.json'
    save_path = args.save_path

    logging.info(f'Train_path: {train_path}')
    logging.info(f'Valid_path: {valid_path}')
    logging.info(f'Test_path: {test_path}')
    logging.info(f'Save_path: {save_path}')
    
    training_data = CivilJPDataset(mode='train', train_file=train_path)
    valid_data = CivilJPDataset(mode='valid', valid_file=valid_path)
    test_data = CivilJPDataset(mode='test', test_file=test_path)

    # 分布式训练sampler：训练时分布式，测试时关闭分布式
    train_sampler = torch.utils.data.distributed.DistributedSampler(training_data)
    #dev_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)
    #test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)

    # pin_memory=True
    # DistributedSampler，需要注意的是在train_loader里面不能再设置shuffle=True
    train_dataloader = DataLoader(
        training_data, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=training_data.collate_function, sampler=train_sampler)
    valid_dataloader = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=valid_data.collate_function)
    test_dataloader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=test_data.collate_function)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # load the model and tokenizer
    if args.model_name in ['KnowAugLawformerCivilJP']:
        model = KnowAugLawformerCivilJP(args=args).cuda(args.local_rank)
    else:
        raise NameError
    
    # 分布式封装模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # 学习率动态下降
    #scheduler = ReduceLROnPlateau(
    #    optimizer, mode='max', factor=0.5, patience=3, verbose=True)  # max for acc

    # 定义SummaryWriter，log_dir是日志文件存储路径
    summary_writer = SummaryWriter(log_dir=args.tensorboard_summary_writer_path)

    # resume checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume_checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # 'cpu' to 'gpu'
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        logging.info(
            f"Resume model and optimizer from checkpoint '{args.resume_checkpoint_path}' with epoch {checkpoint['epoch']} and best FJP Accuracy score of {checkpoint['best_fjp_accuracy_score']}")
        logging.info(f"optimizer lr: {optimizer.param_groups[0]['lr']}")
        start_epoch = checkpoint['epoch']
        best_fjp_accuracy_score = checkpoint['best_fjp_accuracy_score']
    else:
        # start training process
        start_epoch = 0
        best_fjp_accuracy_score = 0
    
    # model.to(device)
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch) # 训练时每一个epoch打乱数据
        model.train()
        start_time = time.time()
        for i, data in enumerate(train_dataloader):
            idx, plaintiff_text, plea_text, defendant_text, fact_text, cause_label_id, gen_article_label_id, spe_article_label_id, fjp_labels_id = data
                        
            # move data to device
            cause_label_id = torch.from_numpy(np.array(cause_label_id)).cuda(args.local_rank, non_blocking=True)
            gen_article_label_id = labels_to_multihot(gen_article_label_id, num_classes=args.gen_article_num_classes)
            gen_article_label_id = torch.from_numpy(np.array(gen_article_label_id)).cuda(args.local_rank, non_blocking=True)
            
            spe_article_label_id = labels_to_multihot(spe_article_label_id, num_classes=args.spe_article_num_classes)
            spe_article_label_id = torch.from_numpy(np.array(spe_article_label_id)).cuda(args.local_rank, non_blocking=True)
            
            # 包含多个诉求的情况，转化成多条样本。
            new_fjp_labels_id = []
            for fjps in fjp_labels_id:
                for fjp in fjps:
                    new_fjp_labels_id.append(fjp)
            
            new_fjp_labels_id = torch.from_numpy(np.array(new_fjp_labels_id)).cuda(args.local_rank, non_blocking=True)

            # forward and backward propagations
            # 全精度更新
            loss, cause_loss, article_loss, fjp_loss, cause_logits, gen_article_logits, spe_article_logits, fjp_logits = model(plaintiff_text, plea_text, defendant_text, fact_text, cause_label_id, gen_article_label_id, spe_article_label_id, new_fjp_labels_id)
            
            # 记录训练集的step loss数据到tensorboard
            summary_writer.add_scalar('train_total_loss', loss.item(), epoch+1)
            summary_writer.add_scalar('train_cause_loss', cause_loss.item(), epoch+1)
            summary_writer.add_scalar('train_article_loss', article_loss.item(), epoch+1)
            summary_writer.add_scalar('train_fjp_loss', fjp_loss.item(), epoch+1)

            # loss regularization, 反向传播
            loss = loss / args.accumulation_steps
            loss.backward()
     
            # 梯度累积
            if ((i+1) % args.accumulation_steps) == 0:
                # optimizer the net
                optimizer.step()

                # zero the parameter gradients
                optimizer.zero_grad()

            if (i + 1) % args.show_per_step == 0:
                # 计算总体所需时间 
                # 计算每个step（i）的耗时
                end_time = time.time()
                per_batch_time = end_time - start_time if i == 0 else (end_time-start_time) / args.show_per_step 
                # 计算预估时间
                est_batch = (len(train_dataloader) - (i+1)) + len(train_dataloader) * (args.epochs - (epoch+1))
                est_time = per_batch_time * est_batch
                start_time = end_time

                # multi-class
                cause_predictions = cause_logits.softmax(dim=1).detach().cpu().numpy()
                cause_label_id = cause_label_id.cpu().numpy()
                cause_pred_label = np.argmax(cause_predictions, axis=1)

                logging.info(f'epoch{epoch+1}, step{i+1:5d}, total_loss: {loss.item():.4f}, cause_loss: {cause_loss.item():.4f}, article_loss: {article_loss.item():.4f}, fjp_loss: {fjp_loss.item():.4f}, Estimated time remaining: {datetime.timedelta(seconds=int(est_time))}')

                cause_accuracy, cause_p_macro, cause_r_macro, cause_f1_macro = get_precision_recall_f1(y_true=cause_label_id, y_pred=cause_pred_label, average='macro')
                _, cause_p_micro, cause_r_micro, cause_f1_micro = get_precision_recall_f1(y_true=cause_label_id, y_pred=cause_pred_label, average='micro')

                logging.info(f'train dataset cause accuracy:{cause_accuracy:.4f} macro-precision:{cause_p_macro:.4f}, macro-recall:{cause_r_macro:.4f}, macro-f1_score:{cause_f1_macro:.4f} || micro-precision:{cause_p_micro:.4f}, micro-recall:{cause_r_micro:.4f}, micro-f1_score:{cause_f1_micro:.4f}')

                # general articles prediction: multi-label
                gen_article_probs = torch.sigmoid(gen_article_logits).detach().cpu().numpy()
                gen_article_pred_label = (gen_article_probs > 0.5).astype(np.float32)
                gen_article_label_id = gen_article_label_id.cpu().numpy()
                
                gen_article_accuracy, gen_article_p_macro, gen_article_r_macro, gen_article_f1_macro = get_precision_recall_f1(y_true=gen_article_label_id, y_pred=gen_article_pred_label, average='macro')
                gen_article_accuracy, gen_article_p_micro, gen_article_r_micro, gen_article_f1_micro = get_precision_recall_f1(y_true=gen_article_label_id, y_pred=gen_article_pred_label, average='micro')
                logging.info(f'Train dataset general articles Accuracy:{gen_article_accuracy:.4f}, macro-p:{gen_article_p_macro:.4f}, macro-r:{gen_article_r_macro:.4f}, macro-f1:{gen_article_f1_macro:.4f} || micro-p:{gen_article_p_micro:.4f}, micro-r:{gen_article_r_micro:.4f}, micro-f1:{gen_article_f1_micro:.4f}')

                # specific articles prediction: multi-label
                spe_article_probs = torch.sigmoid(spe_article_logits).detach().cpu().numpy()
                spe_article_pred_label = (spe_article_probs > 0.5).astype(np.float32)
                spe_article_label_id = spe_article_label_id.cpu().numpy()
                
                spe_article_accuracy, spe_article_p_macro, spe_article_r_macro, spe_article_f1_macro = get_precision_recall_f1(y_true=spe_article_label_id, y_pred=spe_article_pred_label, average='macro')
                spe_article_accuracy, spe_article_p_micro, spe_article_r_micro, spe_article_f1_micro = get_precision_recall_f1(y_true=spe_article_label_id, y_pred=spe_article_pred_label, average='micro')

                logging.info(f'Train dataset specific articles Accuracy: {spe_article_accuracy:.4f}, macro-p: {spe_article_p_macro:.4f}, macro-r: {spe_article_r_macro:.4f}, macro-f1: {spe_article_f1_macro:.4f} || micro-p:{spe_article_p_micro:.4f}, micro-r:{spe_article_r_micro:.4f}, micro-f1:{spe_article_f1_micro:.4f}')

                # fjp: multi-class
                fjp_predictions = fjp_logits.softmax(dim=1).detach().cpu().numpy()
                new_fjp_labels_id = new_fjp_labels_id.cpu().numpy()
                fjp_pred_label = np.argmax(fjp_predictions, axis=1)

                fjp_accuracy, fjp_p_macro, fjp_r_macro, fjp_f1_macro = get_precision_recall_f1(y_true=new_fjp_labels_id, y_pred=fjp_pred_label, average='macro')
                _, fjp_p_micro, fjp_r_micro, fjp_f1_micro = get_precision_recall_f1(y_true=new_fjp_labels_id, y_pred=fjp_pred_label, average='micro')

                logging.info(f'Train dataset fjp accuracy:{fjp_accuracy:.4f} macro-precision:{fjp_p_macro:.4f}, macro-recall:{fjp_r_macro:.4f}, macro-f1_score:{fjp_f1_macro:.4f}||micro-precision:{fjp_p_micro:.4f}, micro-recall:{fjp_r_micro:.4f}, micro-f1_score:{fjp_f1_micro:.4f}')        

        if (epoch + 1) % 1 == 0:
            logging.info('Evaluating the model on validation set...')
            cause_accuracy, cause_p_macro, cause_r_macro, cause_f1_macro, cause_p_micro, cause_r_micro, cause_f1_micro, gen_article_accuracy, gen_article_p_macro, gen_article_r_macro, gen_article_f1_macro, gen_article_p_micro, gen_article_r_micro, gen_article_f1_micro, spe_article_accuracy, spe_article_p_macro, spe_article_r_macro, spe_article_f1_macro, spe_article_p_micro, spe_article_r_micro, spe_article_f1_micro, fjp_accuracy, fjp_p_macro, fjp_r_macro, fjp_f1_macro, fjp_p_micro, fjp_r_micro, fjp_f1_micro = evaluate_civilJP(valid_dataloader, model, device, args)

            logging.info(f'vaild dataset cause accuracy:{cause_accuracy:.4f} macro-precision:{cause_p_macro:.4f}, macro-recall:{cause_r_macro:.4f}, macro-f1_score:{cause_f1_macro:.4f} || micro-precision:{cause_p_micro:.4f}, micro-recall:{cause_r_micro:.4f}, micro-f1_score:{cause_f1_micro:.4f}')
    
            logging.info(f'Valid dataset general articles Accuracy:{gen_article_accuracy:.4f}, macro-p:{gen_article_p_macro:.4f}, macro-r:{gen_article_r_macro:.4f}, macro-f1:{gen_article_f1_macro:.4f} || micro-p:{gen_article_p_micro:.4f}, micro-r:{gen_article_r_micro:.4f}, micro-f1:{gen_article_f1_micro:.4f}')

            logging.info(f'Valid dataset specific articles Accuracy:{spe_article_accuracy:.4f}, macro-p:{spe_article_p_macro:.4f}, macro-r:{spe_article_r_macro:.4f}, macro-f1:{spe_article_f1_macro:.4f} || micro-p:{spe_article_p_micro:.4f}, micro-r:{spe_article_r_micro:.4f}, micro-f1:{spe_article_f1_micro:.4f}')
    
            logging.info(f'Valid dataset fjp accuracy:{fjp_accuracy:.4f} macro-precision:{fjp_p_macro:.4f}, macro-recall:{fjp_r_macro:.4f}, macro-f1_score:{fjp_f1_macro:.4f}||micro-precision:{fjp_p_micro:.4f}, micro-recall:{fjp_r_micro:.4f}, micro-f1_score:{fjp_f1_micro:.4f}')
            
            # 四个子任务的平均f1，其中CCP、FJP使用macro-f1, article使用micro-f1 
            #average_f1_macro = (cause_f1_macro + gen_article_f1_micro + spe_article_f1_micro + fjp_f1_macro) / 4.0
             
            # 用验证集FJP的accuracy来挑选最好的模型 
            if fjp_accuracy > best_fjp_accuracy_score:
                best_fjp_accuracy_score = fjp_accuracy
                logging.info(
                    f"the valid best fjp accuracy score is {best_fjp_accuracy_score}.")
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_fjp_accuracy_score': best_fjp_accuracy_score,
                }
                torch.save(state, save_path)
                logging.info(f'Save model in path: {save_path}')
    
    # 关闭SummaryWriter
    summary_writer.close()
    
    # Attention: Load best checkpoint of dev dataset.
    logging.info('Load best checkpoint for testing model.')
    checkpoint = torch.load(args.save_path, map_location='cpu') # need to check whether use cpu or device. 
    # 分布式训练后，权重会多出一个module，需要将其重命名消除掉才能对齐。
    state_dict = checkpoint['state_dict']
    rename_state_dict = {}
    for k, v in state_dict.items():
        rename_k = k.replace('module.', '') if 'module' in k else k
        rename_state_dict[rename_k] = v
    model.load_state_dict(rename_state_dict)

    # write to excel. Only for test dataset.
    logging.info('Evaluating the model on test set...')
    args.is_metric2csv = True
    cause_accuracy, cause_p_macro, cause_r_macro, cause_f1_macro, cause_p_micro, cause_r_micro, cause_f1_micro, gen_article_accuracy, gen_article_p_macro, gen_article_r_macro, gen_article_f1_macro, gen_article_p_micro, gen_article_r_micro, gen_article_f1_micro, spe_article_accuracy, spe_article_p_macro, spe_article_r_macro, spe_article_f1_macro, spe_article_p_micro, spe_article_r_micro, spe_article_f1_micro, fjp_accuracy, fjp_p_macro, fjp_r_macro, fjp_f1_macro, fjp_p_micro, fjp_r_micro, fjp_f1_micro = evaluate_civilJP(test_dataloader, model, device, args)
    logging.info(f'Test dataset cause accuracy:{cause_accuracy:.4f} macro-precision:{cause_p_macro:.4f}, macro-recall:{cause_r_macro:.4f}, macro-f1_score:{cause_f1_macro:.4f} || micro-precision:{cause_p_micro:.4f}, micro-recall:{cause_r_micro:.4f}, micro-f1_score:{cause_f1_micro:.4f}')

    logging.info(f'Test dataset general articles Accuracy:{gen_article_accuracy:.4f}, macro-p:{gen_article_p_macro:.4f}, macro-r:{gen_article_r_macro:.4f}, macro-f1:{gen_article_f1_macro:.4f} || micro-p:{gen_article_p_micro:.4f}, micro-r:{gen_article_r_micro:.4f}, micro-f1:{gen_article_f1_micro:.4f}')

    logging.info(f'Test dataset specific articles Accuracy:{spe_article_accuracy:.4f}, macro-p:{spe_article_p_macro:.4f}, macro-r:{spe_article_r_macro:.4f}, macro-f1:{spe_article_f1_macro:.4f} || micro-p:{spe_article_p_micro:.4f}, micro-r:{spe_article_r_micro:.4f}, micro-f1:{spe_article_f1_micro:.4f}')

    logging.info(f'Test dataset fjp accuracy:{fjp_accuracy:.4f} macro-precision:{fjp_p_macro:.4f}, macro-recall:{fjp_r_macro:.4f}, macro-f1_score:{fjp_f1_macro:.4f}||micro-precision:{fjp_p_micro:.4f}, micro-recall:{fjp_r_micro:.4f}, micro-f1_score:{fjp_f1_micro:.4f}')