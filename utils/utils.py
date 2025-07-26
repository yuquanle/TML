# -*- encoding: utf-8 -*-
# @author: yuquanle 
# @time: 2023/03/23
# @version: 0.1
# @description:  some utils function.
import csv
import random
import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from transformers import AutoTokenizer


def remove_padding_avg(X, mask):
    """
    function: 移除padding处表征，然后求mean_pooling.
    X: [B, M, h]
    mask: [B, M]
    """
    # 将 padding 位置的特征设置为 0
    X = X * mask.unsqueeze(dim=-1)  # [B, M, h]

    # 计算每个样本移除 padding 后的有效长度
    lengths = mask.sum(dim=1)  # [B]
    
    # 计算每个样本的特征平均
    avg_pool = X.sum(dim=1) / lengths.unsqueeze(dim=-1)  # [B, h]
    
    return avg_pool


def labels_to_multihot(labels, num_classes=146):
    multihot_labels = torch.zeros(len(labels), num_classes)
    for i, label in enumerate(labels):
        for l in label:
            multihot_labels[i][l] = 1
    return multihot_labels


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_precision_recall_f1(y_true: np.array, y_pred: np.array, average='micro'):
    precision = metrics.precision_score(
        y_true, y_pred, average=average, zero_division=0)
    recall = metrics.recall_score(
        y_true, y_pred, average=average, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average=average, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy, precision, recall, f1


def multiClass_metric2csv(y_true, y_pred, csv_path):
    writer = csv.writer(open(csv_path, 'w'))
    writer.writerow(['label', 'precision', 'recall', 'f1-score', 'support'])

    label_metrics = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
 
    dict_prf_acc = {}
    for label in label_metrics:
        if label == 'accuracy':
            dict_prf_acc['accuracy'] = round(label_metrics['accuracy'] * 100, 2)
            continue
        elif label == 'macro avg':
            items = label_metrics[label]
            dict_prf_acc['precision'], dict_prf_acc['recall'], dict_prf_acc['f1-score'] = round(items['precision'] * 100, 2), round(items['recall'] * 100, 2), round(items['f1-score'] * 100, 2)
            
        items = label_metrics[label]
        p, r, f1, support = items['precision'], items['recall'], items['f1-score'], items['support']
        writer.writerow([label, p, r, f1, support])
    
    # the last row save the class average performance.
    writer.writerow(['ACC', 'Macro-P', 'Macro-R', 'Macro-F1'])
    writer.writerow([dict_prf_acc['accuracy'], dict_prf_acc['precision'], dict_prf_acc['recall'], dict_prf_acc['f1-score']])


def multiLabel_metric2csv(y_true, y_pred, csv_path):
    writer = csv.writer(open(csv_path, 'w'))
    writer.writerow(['label', 'precision', 'recall', 'f1-score', 'support'])

    label_metrics = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
 
    dict_prf_acc = {}
    for label in label_metrics:
        if label == 'micro avg':
            items = label_metrics[label]
            dict_prf_acc['micro-precision'], dict_prf_acc['micro-recall'], dict_prf_acc['micro-f1'] = round(items['precision'] * 100, 2), round(items['recall'] * 100, 2), round(items['f1-score'] * 100, 2)
        elif label == 'macro avg':
            items = label_metrics[label]
            dict_prf_acc['macro-precision'], dict_prf_acc['macro-recall'], dict_prf_acc['macro-f1'] = round(items['precision'] * 100, 2), round(items['recall'] * 100, 2), round(items['f1-score'] * 100, 2)

        items = label_metrics[label]
        p, r, f1, support = items['precision'], items['recall'], items['f1-score'], items['support']
        writer.writerow([label, p, r, f1, support])
    
    # the last row save the class average performance.
    writer.writerow(['Micro-P', 'Micro-R', 'Micro-F1'])
    writer.writerow([dict_prf_acc['micro-precision'], dict_prf_acc['micro-recall'], dict_prf_acc['micro-f1']])
    writer.writerow(['Macro-P', 'Macro-R', 'Macro-F1'])
    writer.writerow([dict_prf_acc['macro-precision'], dict_prf_acc['macro-recall'], dict_prf_acc['macro-f1']])


def evaluate_multiclass(test_dataloader, model, args):
    # remove dropout
    model.eval()
    all_predictions = []
    all_labels = []
    tokenizer = AutoTokenizer.from_pretrained(args.PTM_path) 
    for i, data in enumerate(test_dataloader):   
        facts, labels_id = data

        # tokenize the data text
        inputs = tokenizer(list(facts), max_length=args.input_max_length,
                            padding=True, truncation=True, return_tensors='pt')

        # move data to device
        labels_id = torch.from_numpy(np.array(labels_id)).to(args.device)

        # forward 
        with torch.no_grad():
            logits = model(inputs)
        
        all_predictions.append(logits.softmax(dim=1).detach().cpu())
        all_labels.append(labels_id.cpu())

    # concat all sample label or prediction label.
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # write metric to csv file.
    if args.is_metric2csv == True:
        multiClass_metric2csv(all_labels, np.argmax(all_predictions, axis=1), args.metric_save_path)
    
    # macro-p/r/f1
    accuracy, p_macro, r_macro, f1_macro = get_precision_recall_f1(all_labels, np.argmax(all_predictions, axis=1), 'macro')

    # micro-p/r/f1
    _, p_micro, r_micro, f1_micro = get_precision_recall_f1(all_labels, np.argmax(all_predictions, axis=1), 'micro')

    return accuracy, p_macro, r_macro, f1_macro, p_micro, r_micro, f1_micro


def evaluate_HSLCP(test_dataloader, model, args):
    # remove dropout
    model.eval()
    all_predictions = []
    all_labels = []
    tokenizer = AutoTokenizer.from_pretrained(args.PTM_path) 
    for i, data in enumerate(test_dataloader):   
        facts, labels_id = data

        # tokenize the data text
        inputs = tokenizer(list(facts), max_length=args.input_max_length,
                            padding=True, truncation=True, return_tensors='pt')

        # move data to device
        labels_id = torch.from_numpy(np.array(labels_id)).to(args.device)

        # forward 
        with torch.no_grad():
            if args.model_variants in ['Lawformer', 'msBert']:
                logits = model(inputs)
            elif args.model_variants in ['LawformerHS', 'msBertHS']:
                chapter_logits, section_logits, logits = model(inputs)
            else:
                raise NameError
        
        all_predictions.append(logits.softmax(dim=1).detach().cpu())
        all_labels.append(labels_id.cpu())

    # concat all sample label or prediction label.
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # write metric to csv file.
    if args.is_metric2csv == True:
        multiClass_metric2csv(all_labels, np.argmax(all_predictions, axis=1), args.metric_save_path)
    
    # macro-p/r/f1
    accuracy, p_macro, r_macro, f1_macro = get_precision_recall_f1(all_labels, np.argmax(all_predictions, axis=1), 'macro')

    # micro-p/r/f1
    _, p_micro, r_micro, f1_micro = get_precision_recall_f1(all_labels, np.argmax(all_predictions, axis=1), 'micro')

    return accuracy, p_macro, r_macro, f1_macro, p_micro, r_micro, f1_micro
    


def evaluate_Match(test_dataloader, model, args):
    # remove dropout
    model.eval()
    all_predictions = []
    all_labels = []
    for i, data in enumerate(test_dataloader):   
        fact_a, fact_b, labels_id = data

        # move data to device
        labels_id = torch.from_numpy(np.array(labels_id)).to(args.device)

        # forward 
        with torch.no_grad():
            logits = model(fact_a, fact_b)
        
        all_predictions.append(logits.softmax(dim=1).detach().cpu())
        all_labels.append(labels_id.cpu())

    # concat all sample label or prediction label.
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # write metric to csv file.
    if args.is_metric2csv == True:
        multiClass_metric2csv(all_labels, np.argmax(all_predictions, axis=1), args.metric_save_path)
    
    # macro-p/r/f1
    accuracy, p_macro, r_macro, f1_macro = get_precision_recall_f1(all_labels, np.argmax(all_predictions, axis=1), 'macro')

    # micro-p/r/f1
    _, p_micro, r_micro, f1_micro = get_precision_recall_f1(all_labels, np.argmax(all_predictions, axis=1), 'micro')

    return accuracy, p_macro, r_macro, f1_macro, p_micro, r_micro, f1_micro


def evaluate_tripletMatch(test_dataloader, model, args):
    # remove dropout
    model.eval()
    all_predictions = []
    all_labels = []
    tokenizer = AutoTokenizer.from_pretrained(args.PTM_path) 
    for i, data in enumerate(test_dataloader):   
        fact_a, fact_b, fact_c, labels_id, _, _, _ = data

        # move data to device
        labels_id = torch.from_numpy(np.array(labels_id)).to(args.device)

        # forward 
        with torch.no_grad():
            logits = model(fact_a, fact_b, fact_c)
        
        all_predictions.append(logits.softmax(dim=1).detach().cpu())
        all_labels.append(labels_id.cpu())

    # concat all sample label or prediction label.
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # write metric to csv file.
    if args.is_metric2csv == True:
        multiClass_metric2csv(all_labels, np.argmax(all_predictions, axis=1), args.metric_save_path)
    
    # macro-p/r/f1
    accuracy, p_macro, r_macro, f1_macro = get_precision_recall_f1(all_labels, np.argmax(all_predictions, axis=1), 'macro')

    # micro-p/r/f1
    _, p_micro, r_micro, f1_micro = get_precision_recall_f1(all_labels, np.argmax(all_predictions, axis=1), 'micro')

    return accuracy, p_macro, r_macro, f1_macro, p_micro, r_micro, f1_micro
    

def evaluate_KnowAugBert_tripletMatch(test_dataloader, model, args):
    # remove dropout
    model.eval()
    all_predictions = []
    all_labels = []
    tokenizer = AutoTokenizer.from_pretrained(args.PTM_path) 
    for i, data in enumerate(test_dataloader):   
        fact_a, fact_b, fact_c, labels_id, _, _, _, elements_text, elements_indices = data


        # move data to device
        labels_id = torch.from_numpy(np.array(labels_id)).to(args.device)

        # forward 
        with torch.no_grad():
            logits = model(fact_a, fact_b, fact_c, elements_text=elements_text, elements_indices=elements_indices)
        
        all_predictions.append(logits.softmax(dim=1).detach().cpu())
        all_labels.append(labels_id.cpu())

    # concat all sample label or prediction label.
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # write metric to csv file.
    if args.is_metric2csv == True:
        multiClass_metric2csv(all_labels, np.argmax(all_predictions, axis=1), args.metric_save_path)
    
    # macro-p/r/f1
    accuracy, p_macro, r_macro, f1_macro = get_precision_recall_f1(all_labels, np.argmax(all_predictions, axis=1), 'macro')

    # micro-p/r/f1
    _, p_micro, r_micro, f1_micro = get_precision_recall_f1(all_labels, np.argmax(all_predictions, axis=1), 'micro')

    return accuracy, p_macro, r_macro, f1_macro, p_micro, r_micro, f1_micro
    
def evaluate_civilJP_tmp(test_dataloader, model, device, args):
    model.eval()
    all_cause_predictions = []
    all_cause_labels = []
    all_articles_predictions = []
    all_articles_labels = []
    all_fjp_predictions = []
    all_fjp_labels = []
    all_idx = []
    for i, data in enumerate(test_dataloader):       
        # idx: 原样本序号 
        idx, plaintiff, plea, defendant, fact, cause_id, article_id, fjp_id = data

        # move label to device
        articles_label_id = torch.from_numpy(np.array(labels_to_multihot(article_id, num_classes=args.article_num_classes))).to(device)
        cause_label_id = torch.from_numpy(np.array(cause_id)).to(device)
        fjp_label_id = torch.from_numpy(np.array(fjp_id)).to(device)
        idx = torch.from_numpy(np.array(idx)).to(device)

        # forward
        with torch.no_grad():
            cause_logits, article_logits, fjp_logits = model(plaintiff, plea, defendant, fact)
                                                 
        # cause label prediction
        cause_probs = cause_logits.softmax(dim=1).detach().cpu()
        all_cause_predictions.append(cause_probs)
        all_cause_labels.append(cause_label_id.cpu())

        # articles label prediction
        articles_probs = torch.sigmoid(article_logits).detach().cpu()
        all_articles_predictions.append(articles_probs)
        all_articles_labels.append(articles_label_id.cpu())
    
        # fjp label prediction
        fjp_probs = fjp_logits.softmax(dim=1).detach().cpu()
        all_fjp_predictions.append(fjp_probs)
        all_fjp_labels.append(fjp_label_id.cpu())
        
        # 合并测试集的标识
        all_idx.append(idx)
        
    # merge all label prediction
    all_cause_predictions = torch.cat(all_cause_predictions, dim=0).numpy()
    all_articles_predictions = torch.cat(all_articles_predictions, dim=0).numpy()
    all_fjp_predictions = torch.cat(all_fjp_predictions, dim=0).numpy()

    all_cause_labels = torch.cat(all_cause_labels, dim=0).numpy()
    all_articles_labels = torch.cat(all_articles_labels, dim=0).numpy()
    all_fjp_labels = torch.cat(all_fjp_labels, dim=0).numpy()

    all_cause_pred = np.argmax(all_cause_predictions, axis=1)
    all_articles_pred = (all_articles_predictions > 0.5).astype(np.float32)

    all_idx = torch.cat(all_idx, dim=0).cpu().numpy()
    refine_all_cause_labels = []
    refine_all_cause_pred = []
    refine_all_article_labels = []
    refine_all_article_pred = []
    dict_cause_gt_pred = {}
    dict_article_gt_pred = {}
    for idx, gt_cause, pred_cause, gt_article, pred_article in zip(all_idx, all_cause_labels, all_cause_pred, all_articles_labels, all_articles_pred):
        if idx not in dict_cause_gt_pred:
            dict_cause_gt_pred[idx] = [[gt_cause, pred_cause]]
        else:
            dict_cause_gt_pred[idx].append([gt_cause, pred_cause])
        
        if idx not in dict_article_gt_pred:
            dict_article_gt_pred[idx] = [[gt_article, pred_article]]
        else:
            dict_article_gt_pred[idx].append([[gt_article, pred_article]])
    

    # 根据idx排序取对齐标签，只取第一个。TODO:根据投票取
    sorted_dict_cause_gt_pred = dict(sorted(dict_cause_gt_pred.items(), key=lambda x: x[0], reverse=False))
    for key_idx, values_gt_pred in sorted_dict_cause_gt_pred.items():
        refine_all_cause_labels.append(values_gt_pred[0][0])
        refine_all_cause_pred.append(values_gt_pred[0][1])
    
    sorted_dict_article_gt_pred = dict(sorted(dict_article_gt_pred.items(), key=lambda x: x[0], reverse=False))
    for key_idx, values_gt_pred in sorted_dict_article_gt_pred.items():
        refine_all_article_labels.append(values_gt_pred[0][0])
        refine_all_article_pred.append(values_gt_pred[0][1])
    
    assert len(refine_all_article_labels) == len(refine_all_article_pred)
    print(f"Test dataset of cause number is:{len(refine_all_cause_labels)}")
    print(f"Test dataset of article number is:{len(refine_all_article_labels)}")

    # cause prediction
    accuracy_cause, p_macro_cause, r_macro_cause, f1_macro_cause = get_precision_recall_f1(refine_all_cause_labels, refine_all_cause_pred, 'macro')
    accuracy_cause, p_micro_cause, r_micro_cause, f1_micro_cause = get_precision_recall_f1(refine_all_cause_labels, refine_all_cause_pred, 'micro')

    # articles prediction
    accuracy_articles, p_macro_articles, r_macro_articles, f1_macro_articles = get_precision_recall_f1(refine_all_article_labels, refine_all_article_pred, 'macro')
    accuracy_articles, p_micro_articles, r_micro_articles, f1_micro_articles = get_precision_recall_f1(refine_all_article_labels, refine_all_article_pred, 'micro')

    # fjp prediction
    all_fjp_pred = np.argmax(all_fjp_predictions, axis=1)
    accuracy_fjp, p_macro_fjp, r_macro_fjp, f1_macro_fjp = get_precision_recall_f1(all_cause_labels, all_cause_pred, 'macro')
    accuracy_fjp, p_micro_fjp, r_micro_fjp, f1_micro_fjp = get_precision_recall_f1(all_cause_labels, all_cause_pred, 'micro')

    # write each class metric to csv
    if args.is_metric2csv == True:
        multiClass_metric2csv(all_cause_labels, all_cause_pred, args.cause_metric_save_path)
        multiLabel_metric2csv(all_articles_labels, all_articles_pred, args.articles_metric_save_path)
        multiClass_metric2csv(all_fjp_labels, all_fjp_pred, args.fjp_metric_save_path)

    return accuracy_cause, p_macro_cause, r_macro_cause, f1_macro_cause, p_micro_cause, r_micro_cause, f1_micro_cause, accuracy_articles, p_macro_articles, r_macro_articles, f1_macro_articles, p_micro_articles, r_micro_articles, f1_micro_articles, accuracy_fjp, p_macro_fjp, r_macro_fjp, f1_macro_fjp, accuracy_fjp, p_micro_fjp, r_micro_fjp, f1_micro_fjp
