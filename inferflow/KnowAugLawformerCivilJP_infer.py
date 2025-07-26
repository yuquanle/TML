# -*- encoding: utf-8 -*-
# -*- encoding: utf-8 -*-
# @author: yuquanle 
# @time: 2023/06/07
# @version: 0.1
# @description: infer for civil judgment prediction.

import sys
sys.path.append('/home/leyuquan/projects/LegalNLU')
import os
import logging
import numpy as np
import pandas as pd
from utils.utils import get_precision_recall_f1, labels_to_multihot
from utils.CivilJP_utils import evaluate_civilJP, multiClass_metric2csv, multiLabel_metric2csv
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
import csv


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

    # check the device
    device = 'cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu'
    logging.info('Using {} device'.format(device))
    args.device = device
    torch.cuda.empty_cache()

    # prepare test data
    test_path = '../data/'+ args.dataset_name + '/' + args.dataset_name  +'_test.json'
    save_path = args.save_path

    logging.info(f'Test_path: {test_path}')
    logging.info(f'resume_checkpoint_path: {args.resume_checkpoint_path}')
    logging.info(f'model_variants: {args.model_variants}')
    
    test_data = CivilJPDataset(mode='test', test_file=test_path)

    test_dataloader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=test_data.collate_function)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # load the model and tokenizer
    if args.model_name in ['KnowAugLawformerCivilJP']:
        model = KnowAugLawformerCivilJP(args=args)
    else:
        raise NameError

    # resume checkpoint
    checkpoint = torch.load(args.resume_checkpoint_path, map_location='cpu')
    # 分布式训练后，权重会多出一个module，需要将其重命名消除掉才能对齐。
    state_dict = checkpoint['state_dict']
    rename_state_dict = {}
    for k, v in state_dict.items():
        rename_k = k.replace('module.', '') if 'module' in k else k
        rename_state_dict[rename_k] = v
    model.load_state_dict(rename_state_dict)
    
    logging.info(
        f"Resume model and optimizer from checkpoint '{args.resume_checkpoint_path}' with epoch {checkpoint['epoch']} and best FJP Accuracy score of {checkpoint['best_fjp_accuracy_score']}")
   
    model.to(device)
    # write to excel. Only for test dataset.
    logging.info('Evaluating the model on test set...')
    model.eval()
    all_cause_pred_labels = []
    all_cause_gt_labels = []
    all_gen_articles_pred_labels = []
    all_gen_articles_gt_labels = []
    all_spe_articles_pred_labels = []
    all_spe_articles_gt_labels = []
    all_fjp_pred_labels = []
    all_fjp_gt_labels = []
    all_idx = [] #记录案由的编号
    all_fjp_idx = [] #记录fjp切分样本后的案由编号
    all_plaintiff_text = []
    all_plea_text = []
    all_defendant_text = []
    all_fact_text = []
    all_fjp_plaintiff_text = []
    all_fjp_plea_text = []
    all_fjp_defendant_text = []
    all_fjp_fact_text = []
    for i, data in enumerate(test_dataloader):    
        if i % 200 == 0:
            print(f'Processing samples {i * args.batch_size}')   
        # idx: 原样本序号 
        idx, plaintiff_text, plea_text, defendant_text, fact_text, cause_label_id, gen_article_label_id, spe_article_label_id, fjp_labels_id = data
        
        # 案例数据
        all_idx.extend(idx)
        all_plaintiff_text.extend(plaintiff_text)
        all_plea_text.extend(plea_text)
        all_defendant_text.extend(defendant_text)
        all_fact_text.extend(fact_text)

        # move data to device
        cause_label_id = torch.from_numpy(np.array(cause_label_id)).to(device)
        gen_article_label_id = labels_to_multihot(gen_article_label_id, num_classes=args.gen_article_num_classes)
        gen_article_label_id = torch.from_numpy(np.array(gen_article_label_id)).to(device)
        
        spe_article_label_id = labels_to_multihot(spe_article_label_id, num_classes=args.spe_article_num_classes)
        spe_article_label_id = torch.from_numpy(np.array(spe_article_label_id)).to(device)
        
        # 包含多个诉求的情况，转化成多条样本。
        new_fjp_labels_id = []
        for i, fjps, plaintiff, pleas, defendant, fact in zip(idx, fjp_labels_id, plaintiff_text, plea_text, defendant_text, fact_text):
            # 单条样本可能包含多个诉求
            for fjp, plea in zip(fjps, pleas):
                new_fjp_labels_id.append(fjp)
                all_fjp_idx.append(i) 
                all_fjp_plaintiff_text.append(plaintiff)
                all_fjp_plea_text.append(plea)
                all_fjp_defendant_text.append(defendant)
                all_fjp_fact_text.append(fact)
        
        new_fjp_labels_id = torch.from_numpy(np.array(new_fjp_labels_id)).to(device)
        assert len(all_fjp_idx) == len(all_fjp_plea_text)

        # forward
        with torch.no_grad():
            cause_logits, gen_article_logits, spe_article_logits, fjp_logits = model(plaintiff_text, plea_text, defendant_text, fact_text)

        # cause label prediction
        cause_pred_probs = cause_logits.softmax(dim=1).detach().cpu().numpy()
        cause_pred_label = np.argmax(cause_pred_probs, axis=1)
        all_cause_pred_labels.append(torch.from_numpy(cause_pred_label))
        all_cause_gt_labels.append(cause_label_id)

        # gen articles label prediction
        gen_article_probs = torch.sigmoid(gen_article_logits).detach().cpu().numpy()
        gen_article_pred_label = (gen_article_probs > 0.5).astype(np.float32)
        all_gen_articles_pred_labels.append(torch.from_numpy(gen_article_pred_label))
        all_gen_articles_gt_labels.append(gen_article_label_id)
        
        # spe articles label prediction
        spe_article_probs = torch.sigmoid(spe_article_logits).detach().cpu().numpy()
        spe_article_pred_label = (spe_article_probs > 0.5).astype(np.float32)
        all_spe_articles_pred_labels.append(torch.from_numpy(spe_article_pred_label))
        all_spe_articles_gt_labels.append(spe_article_label_id)
        
        # fjp label prediction
        fjp_predictions = fjp_logits.softmax(dim=1).detach().cpu().numpy()
        fjp_pred_label = np.argmax(fjp_predictions, axis=1)
        all_fjp_pred_labels.append(torch.from_numpy(fjp_pred_label))
        all_fjp_gt_labels.append(new_fjp_labels_id)


    # merge all label prediction
    all_cause_pred_labels = torch.cat(all_cause_pred_labels, dim=0).cpu().numpy()
    all_cause_gt_labels = torch.cat(all_cause_gt_labels, dim=0).cpu().numpy()

    all_gen_articles_pred_labels = torch.cat(all_gen_articles_pred_labels, dim=0).cpu().numpy()
    all_gen_articles_gt_labels = torch.cat(all_gen_articles_gt_labels, dim=0).cpu().numpy()

    all_spe_articles_pred_labels = torch.cat(all_spe_articles_pred_labels, dim=0).cpu().numpy()
    all_spe_articles_gt_labels = torch.cat(all_spe_articles_gt_labels, dim=0).cpu().numpy()

    all_fjp_pred_labels = torch.cat(all_fjp_pred_labels, dim=0).cpu().numpy()
    all_fjp_gt_labels = torch.cat(all_fjp_gt_labels, dim=0).cpu().numpy()
    
    
    assert len(all_fjp_pred_labels) == len(all_fjp_gt_labels)
    print(f"Test dataset of cause number is:{len(all_cause_gt_labels)}")
    print(f"Test dataset of gen article number is:{len(all_gen_articles_gt_labels)}")
    print(f"Test dataset of spe cause number is:{len(all_spe_articles_gt_labels)}")
    print(f"Test dataset of fjp number is:{len(all_fjp_gt_labels)}")

    # cause prediction
    cause_accuracy, cause_p_macro, cause_r_macro, cause_f1_macro = get_precision_recall_f1(y_true=all_cause_gt_labels, y_pred=all_cause_pred_labels, average='macro')
    cause_accuracy, cause_p_micro, cause_r_micro, cause_f1_micro = get_precision_recall_f1(y_true=all_cause_gt_labels, y_pred=all_cause_pred_labels, average='micro')

    # general articles prediction
    gen_article_accuracy, gen_article_p_macro, gen_article_r_macro, gen_article_f1_macro = get_precision_recall_f1(y_true=all_gen_articles_gt_labels, y_pred=all_gen_articles_pred_labels, average='macro')
    gen_article_accuracy, gen_article_p_micro, gen_article_r_micro, gen_article_f1_micro = get_precision_recall_f1(y_true=all_gen_articles_gt_labels, y_pred=all_gen_articles_pred_labels, average='micro')

    # specific articles prediction
    spe_article_accuracy, spe_article_p_macro, spe_article_r_macro, spe_article_f1_macro = get_precision_recall_f1(y_true=all_spe_articles_gt_labels, y_pred=all_spe_articles_pred_labels, average='macro')
    spe_article_accuracy, spe_article_p_micro, spe_article_r_micro, spe_article_f1_micro = get_precision_recall_f1(y_true=all_spe_articles_gt_labels, y_pred=all_spe_articles_pred_labels, average='micro')

    # fjp prediction
    fjp_accuracy, fjp_p_macro, fjp_r_macro, fjp_f1_macro = get_precision_recall_f1(y_true=all_fjp_gt_labels, y_pred=all_fjp_pred_labels, average='macro')
    fjp_accuracy, fjp_p_micro, fjp_r_micro, fjp_f1_micro = get_precision_recall_f1(y_true=all_fjp_gt_labels, y_pred=all_fjp_pred_labels, average='micro')

    # write each class metric to csv
    if args.is_metric2csv == True:
        multiClass_metric2csv(all_cause_gt_labels, all_cause_pred_labels, args.cause_metric_save_path)
        multiLabel_metric2csv(all_gen_articles_gt_labels, all_gen_articles_pred_labels, args.gen_articles_metric_save_path)
        multiLabel_metric2csv(all_spe_articles_gt_labels, all_spe_articles_pred_labels, args.spe_articles_metric_save_path)
        multiClass_metric2csv(all_fjp_gt_labels, all_fjp_pred_labels, args.fjp_metric_save_path)

    logging.info(f'Test dataset cause accuracy:{cause_accuracy:.4f} macro-precision:{cause_p_macro:.4f}, macro-recall:{cause_r_macro:.4f}, macro-f1_score:{cause_f1_macro:.4f} || micro-precision:{cause_p_micro:.4f}, micro-recall:{cause_r_micro:.4f}, micro-f1_score:{cause_f1_micro:.4f}')

    logging.info(f'Test dataset general articles Accuracy:{gen_article_accuracy:.4f}, macro-p:{gen_article_p_macro:.4f}, macro-r:{gen_article_r_macro:.4f}, macro-f1:{gen_article_f1_macro:.4f} || micro-p:{gen_article_p_micro:.4f}, micro-r:{gen_article_r_micro:.4f}, micro-f1:{gen_article_f1_micro:.4f}')

    logging.info(f'Test dataset specific articles Accuracy:{spe_article_accuracy:.4f}, macro-p:{spe_article_p_macro:.4f}, macro-r:{spe_article_r_macro:.4f}, macro-f1:{spe_article_f1_macro:.4f} || micro-p:{spe_article_p_micro:.4f}, micro-r:{spe_article_r_micro:.4f}, micro-f1:{spe_article_f1_micro:.4f}')

    logging.info(f'Test dataset fjp accuracy:{fjp_accuracy:.4f} macro-precision:{fjp_p_macro:.4f}, macro-recall:{fjp_r_macro:.4f}, macro-f1_score:{fjp_f1_macro:.4f}||micro-precision:{fjp_p_micro:.4f}, micro-recall:{fjp_r_micro:.4f}, micro-f1_score:{fjp_f1_micro:.4f}')
    
    exit(-1)
    # write law prediction to csv.
    fw_law_pred = csv.writer(open(args.law_pred_save_path, 'w', newline='', encoding='utf-8-sig'))
    fw_law_pred.writerow(['idx', 'claim', 'plea', 'argument', 'fact', 'genlaw_gt', 'genlaw_pred', 'spelaw_gt', 'spelaw_pred'])
   
    for i, claim, plea, argument, fact, genlaw_gt, genlaw_pred, spelaw_gt, spelaw_pred in zip(all_idx, all_plaintiff_text, all_plea_text, all_defendant_text, all_fact_text, all_gen_articles_gt_labels, all_gen_articles_pred_labels, all_spe_articles_gt_labels, all_spe_articles_pred_labels):
        # print(claim, plea, argument, fact, [i for i, x in enumerate(genlaw_gt) if int(x) == int(1)], [i for i, x in enumerate(genlaw_pred) if int(x) == int(1)], [i for i, x in enumerate(spelaw_gt) if int(x) == int(1)], [i for i, x in enumerate(spelaw_pred) if int(x) == int(1)])
        fw_law_pred.writerow([str(i), claim, plea, argument, fact, [i for i, x in enumerate(genlaw_gt) if int(x) == int(1)], [i for i, x in enumerate(genlaw_pred) if int(x) == int(1)], [i for i, x in enumerate(spelaw_gt) if int(x) == int(1)], [i for i, x in enumerate(spelaw_pred) if int(x) == int(1)]])

    # write fjp prediction to csv.
    fw_fjp_pred = csv.writer(open(args.fjp_pred_save_path, 'w', newline='', encoding='utf-8-sig'))
    fw_fjp_pred.writerow(['idx', 'claim', 'plea', 'argument', 'fact', 'fjp_gt', 'fjp_pred'])
    for i, claim, plea, argument, fact, fjp_gt, fjp_pred in zip(all_fjp_idx, all_fjp_plaintiff_text, all_fjp_plea_text, all_fjp_defendant_text, all_fjp_fact_text, all_fjp_gt_labels, all_fjp_pred_labels):
        fw_fjp_pred.writerow([str(i), claim, plea, argument, fact, fjp_gt, fjp_pred])
