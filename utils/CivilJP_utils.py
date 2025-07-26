# -*- encoding: utf-8 -*-
# @author: yuquanle 
# @time: 2023/04/15
# @version: 0.1
# @description: utils for civil judgment prediction.
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import sys
import numpy as np
sys.path.append('/home/leyuquan/projects/LegalNLU')
import torch
import torch.nn.functional as F
from utils.utils import multiClass_metric2csv, multiLabel_metric2csv
from torch.nn.functional import softmax


def remove_diag(matrix):
    '''移除对角线元素
        B, S, S 
    '''
    batch_mat_non_diag = []
    for mat in matrix:
        # remove diagonal elements
        non_diag = mat.flatten()[(torch.arange(mat.shape[0]) != torch.arange(mat.shape[1]).reshape(-1, 1)).flatten()]
        batch_mat_non_diag.append(non_diag.unsqueeze(0))
    return torch.cat(batch_mat_non_diag, dim=0)


def masked_softmax(mask, attention_weights):
    # mask为0的位置表示需要忽略的元素，将其设置为-inf
    attention_weights = attention_weights.masked_fill(mask == 0, -float('inf'))
    # 对于非mask的元素进行softmax
    weights = softmax(attention_weights, dim=-1)
    # 对于mask位置重新设置为0
    weights = weights.masked_fill(torch.isnan(weights), 0)
    return weights


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


def remove_padding_avg_RI(X, mask):
    """
    function: 移除padding处表征，然后求mean_pooling. 考虑全mask的情况，该情况则返回全0特征.
    X: [B, M, h]
    mask: [B, M]
    """
    # 将 attention mask 扩展为 [B, S, M] 的形状
    X = X * mask.unsqueeze(dim=-1)

    # 计算每个样本移除 padding 后的有效长度。可能为全0，防止除0报错。
    lengths = mask.sum(dim=1, keepdim=True).clamp(min=1e-9)

    # 计算每个样本的特征平均
    avg_pool = X.sum(dim=1) / lengths # [B, h]
    
    # 若 mask 中所有元素都为 0，那么 mean pooling 结果为全 0
    avg_pool[lengths.squeeze(1) == 0] = 0

    return avg_pool


def get_base_ccp_inputs(args, tokenizer, inputs_plaintiff, inputs_plea, inputs_defendant):
    '''
    任务一案由预测输入的截断和padding。
    用于单任务lawformer模型。 
    输入格式：[CLS] inputs_plaintiff + inputs_plea + inputs_defendant [SEP]
    参数：
        tokenizer: 模型使用的tokenizer
        inputs_plaintiff: 原告诉称内容
        inputs_plea:  原告诉求内容，可能包含一个或者多个
        inputs_defendant: 被告辩称内容
    '''
    # 案由预测最大序列长度参数
    ccp_input_max_length = args.plaintif_max_length + args.plea_max_length + args.defendant_max_length

    # 初始化输入和掩码
    input_ids = []
    attention_masks = []
    # inputs_plea可能包含多个，在预测案由时，将多个诉求内容拼接到一起
    inputs_plea = [''.join(i) for i in inputs_plea]
    # inputs_plaintiff前面添加[CLS]特殊符号
    inputs_plaintiff = ['[CLS]'+i for i in inputs_plaintiff]
    # inputs_defendant后面添加[SEP]特殊符号。需要考虑如果是被截断，则[SEP]被截断，需要把最后一个token换成[SEP]
    inputs_defendant = [i+'[SEP]' for i in inputs_defendant]

    # 处理 inputs_plaintiff
    if inputs_plaintiff:
        # 对inputs_plaintiff进行编码
        encoded_dict_plaintiff = tokenizer.batch_encode_plus(
                            inputs_plaintiff,                      # 输入的文本
                            add_special_tokens = False,   # 添加 [CLS] 和 [SEP]
                            max_length = args.plaintif_max_length, # 设置最大长度
                            padding='max_length', truncation=True,     # 对输入进行 padding 操作
                            return_attention_mask = True, # 生成 attention mask
                            return_tensors = 'pt'        # 返回 PyTorch 张量类型
                       )
        # 将编码后的文本添加到总输入中
        input_ids.append(encoded_dict_plaintiff['input_ids'])
        attention_masks.append(encoded_dict_plaintiff['attention_mask'])

    # 处理 inputs_plea
    if inputs_plea:
        # 对inputs_plea进行编码
        encoded_dict_plea = tokenizer.batch_encode_plus(
                            inputs_plea,
                            add_special_tokens = False,
                            max_length = args.plea_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_plea['input_ids'])
        attention_masks.append(encoded_dict_plea['attention_mask'])

    # 处理 inputs_defendant
    if inputs_defendant:
        # 对inputs_defendant进行编码
        encoded_dict_defendant = tokenizer.batch_encode_plus(
                            inputs_defendant,
                            add_special_tokens = False,
                            max_length = args.defendant_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_defendant['input_ids'])
        attention_masks.append(encoded_dict_defendant['attention_mask'])

    input_ids = torch.cat(input_ids, dim=1)
    attention_masks = torch.cat(attention_masks, dim=1)
    # 若tensor最后一列不为[PAD]的id（0），说明[SEP]被截断了，则需要把最后一列不为[PAD]的值替换为[SEP]的id（102）
    input_ids[:, -1] = torch.where(input_ids[:, -1] == tokenizer.pad_token_id, input_ids[:, -1], 102)
   
    # input_ids = [CLS] inputs_plaintiff + inputs_plea + inputs_defendant [SEP]
    return input_ids, attention_masks


def get_bert_ccp_inputs(args, tokenizer, inputs_plaintiff, inputs_plea, inputs_defendant, plaintiff_len=None, plea_len=None, defendant_len=None):
    '''
    任务一案由预测输入的截断和padding。
    用于单任务lawformer模型。 
    输入格式：[CLS] inputs_plaintiff + inputs_plea + inputs_defendant [SEP]
    参数：
        tokenizer: 模型使用的tokenizer
        inputs_plaintiff: 原告诉称内容
        inputs_plea:  原告诉求内容，可能包含一个或者多个
        inputs_defendant: 被告辩称内容
    '''
    # 案由预测最大序列长度参数
    ccp_input_max_length = plaintiff_len + plea_len + defendant_len

    assert ccp_input_max_length == 512

    # 初始化输入和掩码
    input_ids = []
    token_type_ids = []
    attention_masks = []
    # inputs_plea可能包含多个，在预测案由时，将多个诉求内容拼接到一起
    inputs_plea = [''.join(i) for i in inputs_plea]
    # inputs_plaintiff前面添加[CLS]特殊符号
    inputs_plaintiff = ['[CLS]'+i for i in inputs_plaintiff]
    # inputs_defendant后面添加[SEP]特殊符号。需要考虑如果是被截断，则[SEP]被截断，需要把最后一个token换成[SEP]
    inputs_defendant = [i+'[SEP]' for i in inputs_defendant]

    # 处理 inputs_plaintiff
    if inputs_plaintiff:
        # 对inputs_plaintiff进行编码
        encoded_dict_plaintiff = tokenizer.batch_encode_plus(
                            inputs_plaintiff,                      # 输入的文本
                            add_special_tokens = False,   # 添加 [CLS] 和 [SEP]
                            max_length = plaintiff_len, # 设置最大长度
                            padding='max_length', truncation=True,     # 对输入进行 padding 操作
                            return_attention_mask = True, # 生成 attention mask
                            return_tensors = 'pt'        # 返回 PyTorch 张量类型
                       )
        # 将编码后的文本添加到总输入中
        input_ids.append(encoded_dict_plaintiff['input_ids'])
        attention_masks.append(encoded_dict_plaintiff['attention_mask'])
        token_type_ids.append(encoded_dict_plaintiff['token_type_ids'])

    # 处理 inputs_plea
    if inputs_plea:
        # 对inputs_plea进行编码
        encoded_dict_plea = tokenizer.batch_encode_plus(
                            inputs_plea,
                            add_special_tokens = False,
                            max_length = plea_len,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_plea['input_ids'])
        attention_masks.append(encoded_dict_plea['attention_mask'])
        token_type_ids.append(encoded_dict_plea['token_type_ids'])

    # 处理 inputs_defendant
    if inputs_defendant:
        # 对inputs_defendant进行编码
        encoded_dict_defendant = tokenizer.batch_encode_plus(
                            inputs_defendant,
                            add_special_tokens = False,
                            max_length = defendant_len,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_defendant['input_ids'])
        attention_masks.append(encoded_dict_defendant['attention_mask'])
        token_type_ids.append(encoded_dict_defendant['token_type_ids'])

    input_ids = torch.cat(input_ids, dim=1)
    attention_masks = torch.cat(attention_masks, dim=1)
    token_type_ids = torch.cat(token_type_ids, dim=1)
    # 若tensor最后一列不为[PAD]的id（0），说明[SEP]被截断了，则需要把最后一列不为[PAD]的值替换为[SEP]的id（102）
    input_ids[:, -1] = torch.where(input_ids[:, -1] == tokenizer.pad_token_id, input_ids[:, -1], 102)
   
    # input_ids = [CLS] inputs_plaintiff + inputs_plea + inputs_defendant [SEP]
    return input_ids, token_type_ids, attention_masks


def get_lawformer_ccp_inputs(args, tokenizer, inputs_plaintiff, inputs_plea, inputs_defendant):
    '''
    任务一案由预测输入的截断和padding。
    用于多任务lawformer模型。 
    输入形式: [CLS][unused1]原告诉称内容[unused2]原告诉请内容[unused3]被告辩称内容[SEP]
    参数：
        tokenizer: 模型使用的tokenizer
        inputs_plaintiff: 原告诉称内容
        inputs_plea:  原告诉求内容，可能包含一个或者多个
        inputs_defendant: 被告辩称内容
    特殊符号表示：
        [unused1]：原告诉称
        [unused2]：原告诉请
        [unused3]：被告辩称
        [unused4]：事实描述
        [unused5]：案由标签描述
        [unused6]：通用法条描述
        [unused7]：特定法条描述
    '''
    # ccp最大长度 = 诉称+诉求+辩称最大长度之和
    ccp_input_max_length = args.plaintif_max_length + args.plea_max_length + args.defendant_max_length 

    # 初始化输入和掩码
    input_ids = []
    attention_masks = []
    
    # inputs_plaintiff: [unused1]表示诉称内容开始符。
    inputs_plaintiff = ['[CLS][unused1]'+i for i in inputs_plaintiff]

    # inputs_plea可能包含多个，在预测案由时，将多个诉求内容拼接到一起
    inputs_plea = [''.join(i) for i in inputs_plea]
    # [unused2]表示原告诉求内容开始符。
    inputs_plea = ['[unused2]'+i for i in inputs_plea] 

    # inputs_defendant后面添加[SEP]特殊符号。需要考虑如果是被截断，则[SEP]被截断，需要把最后一个token换成[SEP]
    # [unused3]：被告辩称内容开始符号
    inputs_defendant = ['[unused3]'+i+'[SEP]' for i in inputs_defendant]
    
    # 处理 inputs_plaintiff
    if inputs_plaintiff:
        # 对inputs_plaintiff进行编码
        encoded_dict_plaintiff = tokenizer.batch_encode_plus(
                            inputs_plaintiff,                      # 输入的文本
                            add_special_tokens = False,   # 添加 [CLS] 和 [SEP]
                            max_length = args.plaintif_max_length, # 设置最大长度
                            padding='max_length', truncation=True,     # 对输入进行 padding 操作
                            return_attention_mask = True, # 生成 attention mask
                            return_tensors = 'pt'        # 返回 PyTorch 张量类型
                       )
        # 将编码后的文本添加到总输入中
        input_ids.append(encoded_dict_plaintiff['input_ids'])
        attention_masks.append(encoded_dict_plaintiff['attention_mask'])

    # 处理 inputs_plea
    if inputs_plea:
        # 对inputs_plea进行编码
        encoded_dict_plea = tokenizer.batch_encode_plus(
                            inputs_plea,
                            add_special_tokens = False,
                            max_length = args.plea_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_plea['input_ids'])
        attention_masks.append(encoded_dict_plea['attention_mask'])

    # 处理 inputs_defendant
    if inputs_defendant:
        # 对inputs_defendant进行编码
        encoded_dict_defendant = tokenizer.batch_encode_plus(
                            inputs_defendant,
                            add_special_tokens = False,
                            max_length = args.defendant_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_defendant['input_ids'])
        attention_masks.append(encoded_dict_defendant['attention_mask'])

    input_ids = torch.cat(input_ids, dim=1)
    attention_masks = torch.cat(attention_masks, dim=1)

    # 若tensor最后一列不为[PAD]的id（0），说明[SEP]被截断了，则需要把最后一列不为[PAD]的值替换为[SEP]的id（102）
    input_ids[:, -1] = torch.where(input_ids[:, -1] == tokenizer.pad_token_id, input_ids[:, -1], 102)
   
    # input_ids = [CLS] [unused1] inputs_plaintiff [unused2] inputs_plea [unused3] inputs_defendant [SEP]
    return input_ids, attention_masks


def get_lawformerCKE_ccp_inputs(args, tokenizer, inputs_plaintiff, inputs_plea, inputs_defendant, 
                               cause_labels_context):
    '''
    Cause knowledge-enhanced lawformer for civil cause prediction.
    任务一案由预测输入的截断和padding。
    输入形式: [CLS][unused5]所有案由标签信息[unused1]原告诉称内容[unused2]原告诉请内容[unused3]被告辩称内容[SEP]
    用于多任务lawformer模型。 
    参数：
        tokenizer: 模型使用的tokenizer
        inputs_plaintiff: 原告诉称内容
        inputs_plea:  原告诉求内容，可能包含一个或者多个
        inputs_defendant: 被告辩称内容
        cause_labels_context：所有的案由标签文字描述内容
    特殊符号表示：
        [unused1] id 1：原告诉称
        [unused2] id 2：原告诉请
        [unused3] id 3：被告辩称
    '''
    # 初始化输入和掩码
    input_ids = []
    attention_masks = []

    # inputs_plaintiff前面添加[CLS]特殊符号: [unused1]表示诉称内容开始符。
    inputs_plaintiff = ['[CLS][unused1]'+i for i in inputs_plaintiff]

    # inputs_plea可能包含多个，在预测案由时，将多个诉求内容拼接到一起
    inputs_plea = [''.join(i) for i in inputs_plea]
    # [unused2]表示原告诉求内容开始符。
    inputs_plea = ['[unused2]'+i for i in inputs_plea] 

    # inputs_defendant：[unused3]为被告辩称内容开始符号
    inputs_defendant = ['[unused3]'+i for i in inputs_defendant]
    
    # 复制batch_size次
    inputs_cause_context = [''.join(cause_labels_context) for _ in range(len(inputs_plaintiff))]
    assert len(inputs_cause_context) == len(inputs_plaintiff)
    # inputs_cause_context：[unused5]表示案由内容开始符，后面添加[SEP]特殊符号作为结尾符。需要考虑如果是被截断，则[SEP]被截断，需要把最后一个token换成[SEP]
    inputs_cause_context = ['[unused5]'+i+'[SEP]' for i in inputs_cause_context]


    # 处理 inputs_plaintiff
    if inputs_plaintiff:
        # 对inputs_plaintiff进行编码
        encoded_dict_plaintiff = tokenizer.batch_encode_plus(
                            inputs_plaintiff,                      # 输入的文本
                            add_special_tokens = False,   # 添加 [CLS] 和 [SEP]
                            max_length = args.plaintif_max_length, # 设置最大长度
                            padding='max_length', truncation=True,     # 对输入进行 padding 操作
                            return_attention_mask = True, # 生成 attention mask
                            return_tensors = 'pt'        # 返回 PyTorch 张量类型
                       )
        # 将编码后的文本添加到总输入中
        input_ids.append(encoded_dict_plaintiff['input_ids'])
        attention_masks.append(encoded_dict_plaintiff['attention_mask'])

    # 处理 inputs_plea
    if inputs_plea:
        # 对inputs_plea进行编码
        encoded_dict_plea = tokenizer.batch_encode_plus(
                            inputs_plea,
                            add_special_tokens = False,
                            max_length = args.plea_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_plea['input_ids'])
        attention_masks.append(encoded_dict_plea['attention_mask'])

    # 处理 inputs_defendant
    if inputs_defendant:
        # 对inputs_defendant进行编码
        encoded_dict_defendant = tokenizer.batch_encode_plus(
                            inputs_defendant,
                            add_special_tokens = False,
                            max_length = args.defendant_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_defendant['input_ids'])
        attention_masks.append(encoded_dict_defendant['attention_mask'])
    
    # 处理 inputs_cause_context
    if inputs_cause_context:
        # 对inputs_cause_context进行编码
        encoded_dict_cause_context = tokenizer.batch_encode_plus(
                            inputs_cause_context,                      # 输入的文本
                            add_special_tokens = False,   # 添加 [CLS] 和 [SEP]
                            max_length = args.ccp_cause_max_length, # 设置最大长度
                            padding='max_length', truncation=True,     # 对输入进行 padding 操作
                            return_attention_mask = True, # 生成 attention mask
                            return_tensors = 'pt'        # 返回 PyTorch 张量类型
                       )
        # 将编码后的文本添加到总输入中
        input_ids.append(encoded_dict_cause_context['input_ids'])
        attention_masks.append(encoded_dict_cause_context['attention_mask'])

    input_ids = torch.cat(input_ids, dim=1)
    attention_masks = torch.cat(attention_masks, dim=1)
    # 若tensor最后一列不为[PAD]的id（0），说明[SEP]被截断了，则需要把最后一列不为[PAD]的值替换为[SEP]的id（102）
    input_ids[:, -1] = torch.where(input_ids[:, -1] == tokenizer.pad_token_id, input_ids[:, -1], 102)
   
    # input_ids = [CLS] [unused1] inputs_plaintiff [unused2] inputs_plea [unused3] inputs_defendant [unused5] cause_context [SEP]

    
    # # 查看位置是否切片正确。
    # # 1表示根据列索引切片，第二个参数表示切片开始位置，第三个参数要切片的长度
    # input_ids_ccp_cause = input_ids.narrow(1, args.plaintif_max_length +  args.plea_max_length +  args.defendant_max_length, args.ccp_cause_max_length)

    # print(f"input_ids:{input_ids}")
    # inputs_cause_length = [len(i) for i in cause_labels_context]
    # inputs_cause_length.insert(0, 0)
    # inputs_cause_start_pos = [sum(inputs_cause_length[:i])+1 for i in range(len(inputs_cause_length))]
    # inputs_cause_length.pop(0)
    # inputs_cause_start_pos.pop(0)
    # for start, length in zip(inputs_cause_start_pos, inputs_cause_length):
    #     print(start, length)
    #     cause_ids = input_ids_ccp_cause.narrow(1, start ,length)
    #     print(f"cause_ids: {cause_ids}")
    # exit(-1)
    return input_ids, attention_masks


def get_lawformer_ccp_MRC_inputs(args, tokenizer, inputs_plaintiff, inputs_plea, inputs_defendant, cause_labels_context, cause_question_prompt):
    '''
    任务一案由预测输入的截断和padding。
    用于多任务lawformer模型。 
    输入形式: [CLS] option [SEP] question [SEP] passage [SEP] 
    具体格式: [CLS] [MASK] 案由1 ... [MASK] 案由N [SEP] prompt_cause [SEP] [unused1] inputs_plaintiff [unused2] inputs_plea [unused3] inputs_defendant [SEP]
    参数：
        tokenizer: 模型使用的tokenizer
        inputs_plaintiff: 原告诉称内容
        inputs_plea:  原告诉求内容，可能包含一个或者多个
        inputs_defendant: 被告辩称内容
        cause_labels_context：所有的案由标签
        cause_question_prompt: 案由的prompt问题
    特殊符号表示：
        [unused1]：原告诉称
        [unused2]：原告诉请
        [unused3]：被告辩称
        [unused4]：事实描述
        [unused5]：案由标签描述
        [unused6]：通用法条描述
        [unused7]：特定法条描述
    '''
    # ccp最大长度 = 诉称+诉求+辩称最大长度之和
    ccp_input_max_length = args.plaintif_max_length + args.plea_max_length + args.defendant_max_length 

    batch_size = len(inputs_plaintiff)

    # 案由标签集合，按照id顺序依次排列。
    # [MASK] cause_label1 [MASK] cause_label2 ...
    cause_labels_context = ''.join([tokenizer.mask_token + cause for cause in cause_labels_context])
    # 复制batch_size次
    inputs_cause_labels_context = [cause_labels_context for _ in range(batch_size)]

    # 初始化输入和掩码
    passage_input_ids = []
    passage_attention_masks = []
    
    # [SEP]用于分隔 option、question、passage内容
    # 1.Option内容：[CLS] [MASK] cause_label1 [MASK] cause_label2... [SEP]，长度固定，因此不需要考虑末尾是否为[SEP]问题。
    inputs_cause_labels_context = ['[CLS]'+i+'[SEP]' for i in inputs_cause_labels_context]

    # 处理 inputs_cause_labels_context
    if inputs_cause_labels_context:
        # 对inputs_cause_labels_context进行编码
        encoded_dict_casue_labels_context = tokenizer.batch_encode_plus(
                            inputs_cause_labels_context,                      # 输入的文本
                            add_special_tokens = False,   # 添加 [CLS] 和 [SEP]
                            max_length = args.ccp_cause_max_length, # 设置最大长度
                            padding='max_length', truncation=True,     # 对输入进行 padding 操作
                            return_attention_mask = True, # 生成 attention mask
                            return_tensors = 'pt'        # 返回 PyTorch 张量类型
                       )
        # 将编码后的文本添加到总输入中
        option_input_ids = encoded_dict_casue_labels_context['input_ids']
        option_attention_masks = encoded_dict_casue_labels_context['attention_mask']
        # print(f"ccp option_input_ids[0]:{option_input_ids[0]}")
        # print(f"ccp option_attention_masks[0]:{option_attention_masks[0]}")


    # 2.Question内容: cause_question_prompt [SEP]，长度固定，因此不需要考虑末尾是否为[SEP]问题。
    inputs_cause_question_prompt = [cause_question_prompt+'[SEP]' for _ in range(batch_size)]

    # 处理 inputs_cause_question_prompt
    if inputs_cause_question_prompt:
        # 对inputs_cause_question_prompt进行编码
        encoded_dict_casue_question_prompt = tokenizer.batch_encode_plus(
                            inputs_cause_question_prompt,                      # 输入的文本
                            add_special_tokens = False,   # 添加 [CLS] 和 [SEP]
                            max_length = args.cause_question_prompt_length, # 设置最大长度
                            padding='max_length', truncation=True,     # 对输入进行 padding 操作
                            return_attention_mask = True, # 生成 attention mask
                            return_tensors = 'pt'        # 返回 PyTorch 张量类型
                       )
        # 将编码后的文本添加到总输入中
        question_input_ids = encoded_dict_casue_question_prompt['input_ids']
        question_attention_masks = encoded_dict_casue_question_prompt['attention_mask']
        # print(f"ccp question_input_ids[0]:{question_input_ids[0]}")
        # print(f"ccp question_attention_masks[0]:{question_attention_masks[0]}")

    # 3.Passage内容: [unused1] inputs_plaintiff [unused2] inputs_plea [unused3] inputs_defendant [SEP]
    # inputs_plaintiff: [unused1]表示诉称内容开始符。
    inputs_plaintiff = ['[unused1]'+i for i in inputs_plaintiff]

    # inputs_plea可能包含多个，在预测案由时，将多个诉求内容拼接到一起
    inputs_plea = [''.join(i) for i in inputs_plea]
    # [unused2]表示原告诉求内容开始符。
    inputs_plea = ['[unused2]'+i for i in inputs_plea] 

    # inputs_defendant后面添加[SEP]特殊符号。需要考虑如果是被截断，则[SEP]被截断，需要把最后一个token换成[SEP]
    # [unused3]：被告辩称内容开始符号
    inputs_defendant = ['[unused3]'+i+'[SEP]' for i in inputs_defendant]
    
    # 处理 inputs_plaintiff
    if inputs_plaintiff:
        # 对inputs_plaintiff进行编码
        encoded_dict_plaintiff = tokenizer.batch_encode_plus(
                            inputs_plaintiff,                      # 输入的文本
                            add_special_tokens = False,   # 添加 [CLS] 和 [SEP]
                            max_length = args.plaintif_max_length, # 设置最大长度
                            padding='max_length', truncation=True,     # 对输入进行 padding 操作
                            return_attention_mask = True, # 生成 attention mask
                            return_tensors = 'pt'        # 返回 PyTorch 张量类型
                       )
        # 将编码后的文本添加到总输入中
        passage_input_ids.append(encoded_dict_plaintiff['input_ids'])
        passage_attention_masks.append(encoded_dict_plaintiff['attention_mask'])

    # 处理 inputs_plea
    if inputs_plea:
        # 对inputs_plea进行编码
        encoded_dict_plea = tokenizer.batch_encode_plus(
                            inputs_plea,
                            add_special_tokens = False,
                            max_length = args.plea_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        passage_input_ids.append(encoded_dict_plea['input_ids'])
        passage_attention_masks.append(encoded_dict_plea['attention_mask'])

    # 处理 inputs_defendant
    if inputs_defendant:
        # 对inputs_defendant进行编码
        encoded_dict_defendant = tokenizer.batch_encode_plus(
                            inputs_defendant,
                            add_special_tokens = False,
                            max_length = args.defendant_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        passage_input_ids.append(encoded_dict_defendant['input_ids'])
        passage_attention_masks.append(encoded_dict_defendant['attention_mask'])

    passage_input_ids = torch.cat(passage_input_ids, dim=1)
    passage_attention_masks = torch.cat(passage_attention_masks, dim=1)

    # 若tensor最后一列不为[PAD]的id（0），说明[SEP]被截断了，则需要把最后一列不为[PAD]的值替换为[SEP]的id（102）
    passage_input_ids[:, -1] = torch.where(passage_input_ids[:, -1] == tokenizer.pad_token_id, passage_input_ids[:, -1], 102)
   
    # print(f"ccp passage_input_ids[0] :{passage_input_ids[0]}")
    # print(f"ccp passage_attention_masks[0]:{passage_attention_masks[0]}")
    
    # 按照顺序拼接option, question, passage.
    input_ids = torch.cat([option_input_ids, question_input_ids, passage_input_ids], dim=1)
    attention_masks = torch.cat([option_attention_masks, question_attention_masks, passage_attention_masks], dim=1)
    # print(f"ccp input_ids[0] :{input_ids[0]}")
    # print(f"ccp attention_masks :{attention_masks}")
    return input_ids, attention_masks

def get_base_clap_inputs(args, tokenizer, inputs_fact):
    '''
    任务二法条预测输入的截断和padding。
    输入格式：[CLS] inputs_fact [SEP]
    用于单任务lawformer模型。
    参数：
        tokenizer: 模型使用的tokenizer
        inputs_fact: 事实描述的内容
    '''
    # 案由预测最大序列长度参数
    clap_input_max_length = args.fact_max_length

    # 对inputs_fact进行编码
    encoded_dict_fact = tokenizer.batch_encode_plus(
                        inputs_fact,
                        add_special_tokens = False,
                        max_length = clap_input_max_length,
                        padding='max_length', truncation=True, 
                        return_attention_mask = True,
                        return_tensors = 'pt'
                    )
    input_ids = encoded_dict_fact['input_ids']
    attention_masks =  encoded_dict_fact['attention_mask']

    # input_ids = [CLS] inputs_fact [SEP]
    return input_ids, attention_masks


def get_bert_clap_inputs(args, tokenizer, inputs_fact, fact_len=None):
    '''
    任务二法条预测输入的截断和padding。
    输入格式：[CLS] inputs_fact [SEP]
    用于bert模型。
    参数：
        tokenizer: 模型使用的tokenizer
        inputs_fact: 事实描述的内容
    '''
    # 案由预测最大序列长度参数
    clap_input_max_length = fact_len
    assert clap_input_max_length == 512

    # 对inputs_fact进行编码
    encoded_dict_fact = tokenizer.batch_encode_plus(
                        inputs_fact,
                        add_special_tokens = False,
                        max_length = clap_input_max_length,
                        padding='max_length', truncation=True, 
                        return_attention_mask = True,
                        return_tensors = 'pt'
                    )
    input_ids = encoded_dict_fact['input_ids']
    token_type_ids = encoded_dict_fact['token_type_ids']
    attention_masks =  encoded_dict_fact['attention_mask']

    # input_ids = [CLS] inputs_fact [SEP]
    return input_ids, token_type_ids, attention_masks


def get_lawformer_clap_inputs(args, tokenizer, inputs_fact):
    '''
    任务二法条预测输入的截断和padding。
    输入格式：[CLS] [unused4] inputs_fact [SEP]
    用于多任务lawformer模型。
    参数：
        tokenizer: 模型使用的tokenizer
        inputs_fact: 事实描述的内容
    特殊符号表示：
        [unused4]：事实描述
    '''
    # 案由预测最大序列长度参数
    clap_input_max_length = args.fact_max_length

    inputs_fact = ['[CLS][unused4]'+i+'[SEP]' for i in inputs_fact]

    # 对inputs_fact进行编码
    encoded_dict_fact = tokenizer.batch_encode_plus(
                        inputs_fact,
                        add_special_tokens = False,
                        max_length = clap_input_max_length,
                        padding='max_length', truncation=True, 
                        return_attention_mask = True,
                        return_tensors = 'pt'
                    )
    input_ids = encoded_dict_fact['input_ids']
    attention_masks =  encoded_dict_fact['attention_mask']

    # 若tensor最后一列不为[PAD]的id（0），说明[SEP]被截断了，则需要把最后一列不为[PAD]的值替换为[SEP]的id（102）
    input_ids[:, -1] = torch.where(input_ids[:, -1] == tokenizer.pad_token_id, input_ids[:, -1], 102)
   
    # input_ids = [CLS][unused4] inputs_fact [SEP]
    return input_ids, attention_masks


def get_lawformerTCKE_clap_inputs(args, tokenizer, inputs_fact, cause_labels_context, cause_logits):
    '''
    任务二法条预测输入的截断和padding。
    输入格式：[CLS] [unused4] inputs_fact [unused5]top-K预测的案由标签信息 [SEP]
    用于多任务lawformer模型。
    参数：
        tokenizer: 模型使用的tokenizer
        inputs_fact: 事实描述的内容
        cause_labels_context：所有的案由标签文字描述内容
        cause_logits: 案由的预测logits
    特殊符号表示：
        [unused4]：事实描述
        [unused5]：案由标签信息
    '''
    input_ids = []
    attention_masks = []

    inputs_fact = ['[CLS][unused4]'+i for i in inputs_fact]

    # 取cause_logits的top-k案由标签预测
    # 获取每行最大的两个值以及对应的位置
    cause_top_k_values, cause_top_k_indices = torch.topk(cause_logits, args.cause_top_k, dim=1)

    inputs_topk_cause_context = []
    for top_k_indice in cause_top_k_indices:
        sample_cause_context = ['[unused5]']
        for indice in top_k_indice:
            sample_cause_context.append(cause_labels_context[indice])
        sample_cause_context.append('[SEP]') 
        inputs_topk_cause_context.append(''.join(sample_cause_context))
    
    if inputs_fact:
        # 对inputs_fact进行编码
        encoded_dict_fact = tokenizer.batch_encode_plus(
                            inputs_fact,
                            add_special_tokens = False,
                            max_length = args.fact_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                        )
        input_ids.append(encoded_dict_fact['input_ids'])
        attention_masks.append(encoded_dict_fact['attention_mask'])

    # 处理 inputs_cause_context
    if inputs_topk_cause_context:
        # 对inputs_cause_context进行编码
        encoded_dict_topk_cause_context = tokenizer.batch_encode_plus(
                            inputs_topk_cause_context,
                            add_special_tokens = False,
                            max_length = args.topk_cause_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_topk_cause_context['input_ids'])
        attention_masks.append(encoded_dict_topk_cause_context['attention_mask'])

    input_ids = torch.cat(input_ids, dim=1)
    attention_masks = torch.cat(attention_masks, dim=1)

    # 若tensor最后一列不为[PAD]的id（0），说明[SEP]被截断了，则需要把最后一列不为[PAD]的值替换为[SEP]的id（102）
    input_ids[:, -1] = torch.where(input_ids[:, -1] == tokenizer.pad_token_id, input_ids[:, -1], 102)
   
    # input_ids = [CLS] [unused4] inputs_fact [unused5] top-k pred cause contexts [SEP]
    return input_ids, attention_masks


def get_lawformer_clap_MRC_inputs(args, tokenizer, inputs_fact, article_name, article_question_prompt, all_article_label_max_length, article_question_prompt_max_length):
    '''
    任务二法条预测输入的截断和padding。
    # 任务二输入格式: [CLS] option [SEP] question [SEP] passage [SEP] 
    # 具体输入格式：[CLS] [MASK] 法条1 ... [MASK] 法条N [SEP] article_question_prompt [SEP] [unused4] inputs_fact [SEP]
    用于多任务lawformer_MRC模型。
    参数：
        tokenizer: 模型使用的tokenizer
        inputs_fact: 事实描述的内容
        all_article_label_max_length: 所有法条名字的输入最大长度
    特殊符号表示：
        [unused4]：事实描述
    '''
    batch_size = len(inputs_fact)
    # 案由预测最大序列长度参数
    clap_input_max_length = args.fact_max_length

    # 法条标签集合，按照id顺序依次排列。
    # [MASK] article_label1 [MASK] article_label2 ...
    all_article_labels_name = ''.join([tokenizer.mask_token + article for article in article_name])
    # 复制batch_size次
    inputs_all_article_labels_name = [all_article_labels_name for _ in range(batch_size)]


    # [SEP]用于分隔 option、question、passage内容
    # 1.Option内容：[MASK] article_label1 [MASK] article_label2 ...，长度固定，因此不需要考虑末尾是否为[SEP]问题。
    inputs_all_article_labels_name = ['[CLS]'+i+'[SEP]' for i in inputs_all_article_labels_name]

    # 处理 inputs_all_article_labels_name
    if inputs_all_article_labels_name:
        encoded_dict_all_article_labels_name = tokenizer.batch_encode_plus(
                            inputs_all_article_labels_name,                      # 输入的文本
                            add_special_tokens = False,   # 添加 [CLS] 和 [SEP]
                            max_length = all_article_label_max_length, # 设置最大长度
                            padding='max_length', truncation=True,     # 对输入进行 padding 操作
                            return_attention_mask = True, # 生成 attention mask
                            return_tensors = 'pt'        # 返回 PyTorch 张量类型
                       )
        # 将编码后的文本添加到总输入中
        option_input_ids = encoded_dict_all_article_labels_name['input_ids']
        option_attention_masks = encoded_dict_all_article_labels_name['attention_mask']
        # print(f"clap option_input_ids[0]: {option_input_ids[0]}")
        # print(f"clap option_attention_masks[0]: {option_attention_masks[0]}")


    # 2.Question内容: article_question_prompt [SEP]，长度固定，因此不需要考虑末尾是否为[SEP]问题。
    inputs_article_question_prompt = [article_question_prompt+'[SEP]' for _ in range(batch_size)]

    if inputs_article_question_prompt:
        encoded_dict_article_question_prompt = tokenizer.batch_encode_plus(
                            inputs_article_question_prompt,                      # 输入的文本
                            add_special_tokens = False,   # 添加 [CLS] 和 [SEP]
                            max_length = article_question_prompt_max_length, # 设置最大长度
                            padding='max_length', truncation=True,     # 对输入进行 padding 操作
                            return_attention_mask = True, # 生成 attention mask
                            return_tensors = 'pt'        # 返回 PyTorch 张量类型
                       )
        # 将编码后的文本添加到总输入中
        question_input_ids = encoded_dict_article_question_prompt['input_ids']
        question_attention_masks = encoded_dict_article_question_prompt['attention_mask']
        # print(f"clap question_input_ids[0]: {question_input_ids[0]}")
        # print(f"clap question_attention_masks[0]: {question_attention_masks[0]}")


    # 3.Passage内容: [unused4] fact [SEP]
    inputs_fact = ['[unused4]'+i+'[SEP]' for i in inputs_fact]

    # 对inputs_fact进行编码
    encoded_dict_fact = tokenizer.batch_encode_plus(
                        inputs_fact,
                        add_special_tokens = False,
                        max_length = args.fact_max_length,
                        padding='max_length', truncation=True, 
                        return_attention_mask = True,
                        return_tensors = 'pt'
                    )
    passage_input_ids = encoded_dict_fact['input_ids']
    passage_attention_masks =  encoded_dict_fact['attention_mask']

    # 若tensor最后一列不为[PAD]的id（0），说明[SEP]被截断了，则需要把最后一列不为[PAD]的值替换为[SEP]的id（102）
    passage_input_ids[:, -1] = torch.where(passage_input_ids[:, -1] == tokenizer.pad_token_id, passage_input_ids[:, -1], 102)

    # print(f"clap passage_input_ids[0]: {passage_input_ids[0]}")
    # print(f"clap passage_attention_masks[0]: {passage_attention_masks[0]}")

    # 按照顺序拼接option, question, passage.
    input_ids = torch.cat([option_input_ids, question_input_ids, passage_input_ids], dim=1)
    attention_masks = torch.cat([option_attention_masks, question_attention_masks, passage_attention_masks], dim=1)
    # print(f"clap input_ids[0]: {list(input_ids[0].cpu().numpy())}")
    # print(f"clap attention_masks[0]: {list(attention_masks[0].cpu().numpy())}")
    return input_ids, attention_masks


def get_lawformerTCKE_clap_MRC_inputs(args, tokenizer, inputs_fact, article_name, article_question_prompt, all_article_label_max_length, article_question_prompt_max_length, cause_labels_name, cause_logits):
    '''
    任务二法条预测输入的截断和padding。
    # 任务二输入格式: [CLS] option [SEP] question [SEP] passage [SEP] 
    # 具体输入格式：[CLS] [MASK] 法条1 ... [MASK] 法条N [SEP] article_question_prompt [SEP] [unused4] inputs_fact [unused5] top-K预测的案由标签信息 [SEP]
    用于多任务lawformer_MRC模型。
    参数：
        tokenizer: 模型使用的tokenizer
        inputs_fact: 事实描述的内容
        all_article_label_max_length: 所有法条名字的输入最大长度
        cause_name：所有案由名
        cause_logits: 案由预测概率
    特殊符号表示：
        [unused4]：事实描述
    '''
    # 取cause_logits的top-k案由标签预测
    # 获取每行最大的两个值以及对应的位置
    cause_top_k_values, cause_top_k_indices = torch.topk(cause_logits, args.cause_top_k, dim=1)

    inputs_topk_cause_label_name = []
    for top_k_indice in cause_top_k_indices:
        sample_cause_context = ['[unused5]']
        for indice in top_k_indice:
            sample_cause_context.append(cause_labels_name[indice])
        sample_cause_context.append('[SEP]') 
        inputs_topk_cause_label_name.append(''.join(sample_cause_context))

    batch_size = len(inputs_fact)
    # 法条标签集合，按照id顺序依次排列。
    # [MASK] article_label1 [MASK] article_label2 ...
    all_article_labels_name = ''.join([tokenizer.mask_token + article for article in article_name])
    # 复制batch_size次
    inputs_all_article_labels_name = [all_article_labels_name for _ in range(batch_size)]


    # [SEP]用于分隔 option、question、passage内容
    # 1.Option内容：[MASK] article_label1 [MASK] article_label2 ...，长度固定，因此不需要考虑末尾是否为[SEP]问题。
    inputs_all_article_labels_name = ['[CLS]'+i+'[SEP]' for i in inputs_all_article_labels_name]

    # 处理 inputs_all_article_labels_name
    if inputs_all_article_labels_name:
        encoded_dict_all_article_labels_name = tokenizer.batch_encode_plus(
                            inputs_all_article_labels_name,                      # 输入的文本
                            add_special_tokens = False,   # 添加 [CLS] 和 [SEP]
                            max_length = all_article_label_max_length, # 设置最大长度
                            padding='max_length', truncation=True,     # 对输入进行 padding 操作
                            return_attention_mask = True, # 生成 attention mask
                            return_tensors = 'pt'        # 返回 PyTorch 张量类型
                       )
        # 将编码后的文本添加到总输入中
        option_input_ids = encoded_dict_all_article_labels_name['input_ids']
        option_attention_masks = encoded_dict_all_article_labels_name['attention_mask']
        # print(f"clap option_input_ids[0]: {option_input_ids[0]}")
        # print(f"clap option_attention_masks[0]: {option_attention_masks[0]}")


    # 2.Question内容: article_question_prompt [SEP]，长度固定，因此不需要考虑末尾是否为[SEP]问题。
    inputs_article_question_prompt = [article_question_prompt+'[SEP]' for _ in range(batch_size)]

    if inputs_article_question_prompt:
        encoded_dict_article_question_prompt = tokenizer.batch_encode_plus(
                            inputs_article_question_prompt,                      # 输入的文本
                            add_special_tokens = False,   # 添加 [CLS] 和 [SEP]
                            max_length = article_question_prompt_max_length, # 设置最大长度
                            padding='max_length', truncation=True,     # 对输入进行 padding 操作
                            return_attention_mask = True, # 生成 attention mask
                            return_tensors = 'pt'        # 返回 PyTorch 张量类型
                       )
        # 将编码后的文本添加到总输入中
        question_input_ids = encoded_dict_article_question_prompt['input_ids']
        question_attention_masks = encoded_dict_article_question_prompt['attention_mask']
        # print(f"clap question_input_ids[0]: {question_input_ids[0]}")
        # print(f"clap question_attention_masks[0]: {question_attention_masks[0]}")


    # 3.Passage内容: [unused4] fact [SEP]
    passage_input_ids = []
    passage_attention_masks = []

    inputs_fact = ['[unused4]'+i+'[SEP]' for i in inputs_fact]

    # 对inputs_fact进行编码
    encoded_dict_fact = tokenizer.batch_encode_plus(
                        inputs_fact,
                        add_special_tokens = False,
                        max_length = args.fact_max_length,
                        padding='max_length', truncation=True, 
                        return_attention_mask = True,
                        return_tensors = 'pt'
                    )
    passage_input_ids.append(encoded_dict_fact['input_ids'])
    passage_attention_masks.append(encoded_dict_fact['attention_mask'])

    # 对inputs_topk_cause_label_name进行编码
    encoded_dict_topk_cause_label_name = tokenizer.batch_encode_plus(
                        inputs_topk_cause_label_name,
                        add_special_tokens = False,
                        max_length = args.topk_cause_max_length,
                        padding='max_length', truncation=True, 
                        return_attention_mask = True,
                        return_tensors = 'pt'
                    )
    passage_input_ids.append(encoded_dict_topk_cause_label_name['input_ids'])
    passage_attention_masks.append(encoded_dict_topk_cause_label_name['attention_mask'])

    passage_input_ids = torch.cat(passage_input_ids, dim=1)
    passage_attention_masks = torch.cat(passage_attention_masks, dim=1)

    # 若tensor最后一列不为[PAD]的id（0），说明[SEP]被截断了，则需要把最后一列不为[PAD]的值替换为[SEP]的id（102）
    passage_input_ids[:, -1] = torch.where(passage_input_ids[:, -1] == tokenizer.pad_token_id, passage_input_ids[:, -1], 102)

    # print(f"clap passage_input_ids[0]: {passage_input_ids[0]}")
    # print(f"clap passage_attention_masks[0]: {passage_attention_masks[0]}")

    # 按照顺序拼接option, question, passage.
    input_ids = torch.cat([option_input_ids, question_input_ids, passage_input_ids], dim=1)
    attention_masks = torch.cat([option_attention_masks, question_attention_masks, passage_attention_masks], dim=1)
    # print(f"clap input_ids[0]: {list(input_ids[0].cpu().numpy())}")
    # print(f"clap attention_masks[0]: {list(attention_masks[0].cpu().numpy())}")
    return input_ids, attention_masks


def get_base_fjp_inputs(args, tokenizer, inputs_plaintiff, inputs_plea, inputs_defendant, inputs_fact):
    '''
    任务三案由预测输入的截断和padding。
    输入格式：[CLS] inputs_fact + inputs_plea + inputs_defendant + inputs_fact [SEP]
    用于单任务lawformer模型。
    参数：
        tokenizer: 模型使用的tokenizer
        inputs_plaintiff: 原告诉称内容
        inputs_plea:  原告诉求内容，可能包含一个或者多个，如果是多个，则需要拆分，因此batch_size会发生变化。
        inputs_defendant: 被告辩称内容
        inputs_fact: 事实描述的内容
    '''
    # 初始化输入和掩码
    input_ids = []
    attention_masks = []
    
    # 最大长度断言
    assert args.input_max_length == args.plaintif_max_length + args.plea_max_length + args.defendant_max_length + args.fact_max_length
    
    # 根据诉求数量重组数据
    new_inputs_plaintiff = []
    new_inputs_plea = []
    new_inputs_defendant = []
    new_inputs_fact = []
    for plaintiff, pleas, defendant, fact in zip(inputs_plaintiff, inputs_plea, inputs_defendant, inputs_fact):
        # 单条样本可能包含多个诉求
        for plea in pleas:
            new_inputs_plaintiff.append(plaintiff)
            new_inputs_plea.append(plea)
            new_inputs_defendant.append(defendant)
            new_inputs_fact.append(fact)
    assert len(new_inputs_plaintiff) == len(new_inputs_plea)

    # new_inputs_plaintiff前面添加[CLS]特殊符号
    new_inputs_plaintiff = ['[CLS]'+i for i in new_inputs_plaintiff]
    # inputs_fact后面添加[SEP]特殊符号。需要考虑如果是被截断，则[SEP]被截断，需要把最后一个token换成[SEP]
    new_inputs_fact = [i+'[SEP]' for i in new_inputs_fact]

    # 处理 new_inputs_plaintiff
    if new_inputs_plaintiff:
        # 对new_inputs_plaintiff进行编码
        encoded_dict_plaintiff = tokenizer.batch_encode_plus(
                            new_inputs_plaintiff,                      # 输入的文本
                            add_special_tokens = False,   # 添加 [CLS] 和 [SEP]
                            max_length = args.plaintif_max_length, # 设置最大长度
                            padding='max_length', truncation=True,     # 对输入进行 padding 操作
                            return_attention_mask = True, # 生成 attention mask
                            return_tensors = 'pt'        # 返回 PyTorch 张量类型
                       )
        # 将编码后的文本添加到总输入中
        input_ids.append(encoded_dict_plaintiff['input_ids'])
        attention_masks.append(encoded_dict_plaintiff['attention_mask'])

    # 处理 new_inputs_plea
    if new_inputs_plea:
        # 对new_inputs_plea进行编码
        encoded_dict_plea = tokenizer.batch_encode_plus(
                            new_inputs_plea,
                            add_special_tokens = False,
                            max_length = args.plea_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_plea['input_ids'])
        attention_masks.append(encoded_dict_plea['attention_mask'])

    # 处理 new_inputs_defendant
    if new_inputs_defendant:
        # 对new_inputs_defendant进行编码
        encoded_dict_defendant = tokenizer.batch_encode_plus(
                            new_inputs_defendant,
                            add_special_tokens = False,
                            max_length = args.defendant_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_defendant['input_ids'])
        attention_masks.append(encoded_dict_defendant['attention_mask'])

    # 处理 new_inputs_fact
    if new_inputs_fact:
        # 对new_inputs_fact进行编码
        encoded_dict_fact = tokenizer.batch_encode_plus(
                            new_inputs_fact,
                            add_special_tokens = False,
                            max_length = args.fact_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_fact['input_ids'])
        attention_masks.append(encoded_dict_fact['attention_mask'])

    input_ids = torch.cat(input_ids, dim=1)
    attention_masks = torch.cat(attention_masks, dim=1)
    # 若tensor最后一列不为[PAD]的id（0），说明[SEP]被截断了，则需要把最后一列不为[PAD]的值替换为[SEP]的id（102）
    input_ids[:, -1] = torch.where(input_ids[:, -1] == tokenizer.pad_token_id, input_ids[:, -1], 102)
   
    # input_ids = [CLS] inputs_plaintiff + 单条inputs_plea + inputs_defendant + inputs_fact [SEP]
    return input_ids, attention_masks


def get_bert_fjp_inputs(args, tokenizer, inputs_plaintiff, inputs_plea, inputs_defendant, inputs_fact, plaintiff_len=None, plea_len=None, defendant_len=None, fact_len=None):
    '''
    任务三案由预测输入的截断和padding。
    输入格式：[CLS] inputs_fact + inputs_plea + inputs_defendant + inputs_fact [SEP]
    用于bert模型。
    参数：
        tokenizer: 模型使用的tokenizer
        inputs_plaintiff: 原告诉称内容
        inputs_plea:  原告诉求内容，可能包含一个或者多个，如果是多个，则需要拆分，因此batch_size会发生变化。
        inputs_defendant: 被告辩称内容
        inputs_fact: 事实描述的内容
    '''
    # 初始化输入和掩码
    input_ids = []
    token_type_ids = []
    attention_masks = []
    
    # 最大长度断言
    assert 512 == plaintiff_len + plea_len + defendant_len + fact_len
    
    # 根据诉求数量重组数据
    new_inputs_plaintiff = []
    new_inputs_plea = []
    new_inputs_defendant = []
    new_inputs_fact = []
    for plaintiff, pleas, defendant, fact in zip(inputs_plaintiff, inputs_plea, inputs_defendant, inputs_fact):
        # 单条样本可能包含多个诉求
        for plea in pleas:
            new_inputs_plaintiff.append(plaintiff)
            new_inputs_plea.append(plea)
            new_inputs_defendant.append(defendant)
            new_inputs_fact.append(fact)
    assert len(new_inputs_plaintiff) == len(new_inputs_plea)

    # new_inputs_plaintiff前面添加[CLS]特殊符号
    new_inputs_plaintiff = ['[CLS]'+i for i in new_inputs_plaintiff]
    # inputs_fact后面添加[SEP]特殊符号。需要考虑如果是被截断，则[SEP]被截断，需要把最后一个token换成[SEP]
    new_inputs_fact = [i+'[SEP]' for i in new_inputs_fact]

    # 处理 new_inputs_plaintiff
    if new_inputs_plaintiff:
        # 对new_inputs_plaintiff进行编码
        encoded_dict_plaintiff = tokenizer.batch_encode_plus(
                            new_inputs_plaintiff,                      # 输入的文本
                            add_special_tokens = False,   # 添加 [CLS] 和 [SEP]
                            max_length = plaintiff_len, # 设置最大长度
                            padding='max_length', truncation=True,     # 对输入进行 padding 操作
                            return_attention_mask = True, # 生成 attention mask
                            return_tensors = 'pt'        # 返回 PyTorch 张量类型
                       )
        # 将编码后的文本添加到总输入中
        input_ids.append(encoded_dict_plaintiff['input_ids'])
        token_type_ids.append(encoded_dict_plaintiff['token_type_ids'])
        attention_masks.append(encoded_dict_plaintiff['attention_mask'])

    # 处理 new_inputs_plea
    if new_inputs_plea:
        # 对new_inputs_plea进行编码
        encoded_dict_plea = tokenizer.batch_encode_plus(
                            new_inputs_plea,
                            add_special_tokens = False,
                            max_length = plea_len,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_plea['input_ids'])
        token_type_ids.append(encoded_dict_plea['token_type_ids'])
        attention_masks.append(encoded_dict_plea['attention_mask'])

    # 处理 new_inputs_defendant
    if new_inputs_defendant:
        # 对new_inputs_defendant进行编码
        encoded_dict_defendant = tokenizer.batch_encode_plus(
                            new_inputs_defendant,
                            add_special_tokens = False,
                            max_length = defendant_len,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_defendant['input_ids'])
        token_type_ids.append(encoded_dict_defendant['token_type_ids'])
        attention_masks.append(encoded_dict_defendant['attention_mask'])

    # 处理 new_inputs_fact
    if new_inputs_fact:
        # 对new_inputs_fact进行编码
        encoded_dict_fact = tokenizer.batch_encode_plus(
                            new_inputs_fact,
                            add_special_tokens = False,
                            max_length = fact_len,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_fact['input_ids'])
        token_type_ids.append(encoded_dict_fact['token_type_ids'])
        attention_masks.append(encoded_dict_fact['attention_mask'])

    input_ids = torch.cat(input_ids, dim=1)
    token_type_ids = torch.cat(token_type_ids, dim=1)
    attention_masks = torch.cat(attention_masks, dim=1)
    # 若tensor最后一列不为[PAD]的id（0），说明[SEP]被截断了，则需要把最后一列不为[PAD]的值替换为[SEP]的id（102）
    input_ids[:, -1] = torch.where(input_ids[:, -1] == tokenizer.pad_token_id, input_ids[:, -1], 102)
   
    # input_ids = [CLS] inputs_plaintiff + 单条inputs_plea + inputs_defendant + inputs_fact [SEP]
    return input_ids, token_type_ids, attention_masks


def get_lawformer_fjp_inputs(args, tokenizer, inputs_plaintiff, inputs_plea, inputs_defendant, inputs_fact):
    '''
    任务三案由预测输入的截断和padding。
    输入格式：[CLS] [unused1] inputs_plaintiff [unused2] inputs_plea [unused3] inputs_defendant [unused4] inputs_fact [SEP]
    用于多任务lawformer、lawformer_CKE、lawformer_TCKE模型。
    参数：
        tokenizer: 模型使用的tokenizer
        inputs_plaintiff: 原告诉称内容
        inputs_plea:  原告诉求内容，可能包含一个或者多个，如果是多个，则需要拆分，因此batch_size会发生变化。
        inputs_defendant: 被告辩称内容
        inputs_fact: 事实描述的内容
    特殊符号表示：
        [unused1]：原告诉称
        [unused2]：原告诉请
        [unused3]：被告辩称
        [unused4]：事实描述
        [unused5]：案由标签描述
        [unused6]：通用法条描述
        [unused7]：特定法条描述
    '''
    # 初始化输入和掩码
    input_ids = []
    attention_masks = []
    
    # 最大长度断言
    assert args.input_max_length == args.plaintif_max_length + args.plea_max_length + args.defendant_max_length + args.fact_max_length
    
    # 根据诉求数量重组数据
    new_inputs_plaintiff = []
    new_inputs_plea = []
    new_inputs_defendant = []
    new_inputs_fact = []
    for plaintiff, pleas, defendant, fact in zip(inputs_plaintiff, inputs_plea, inputs_defendant, inputs_fact):
        # 单条样本可能包含多个诉求
        for plea in pleas:
            new_inputs_plaintiff.append(plaintiff)
            new_inputs_plea.append(plea)
            new_inputs_defendant.append(defendant)
            new_inputs_fact.append(fact)
    assert len(new_inputs_plaintiff) == len(new_inputs_plea)

    # new_inputs_plaintiff前面添加[CLS]特殊符号
    new_inputs_plaintiff = ['[CLS][unused1]'+i for i in new_inputs_plaintiff]
    new_inputs_plea = ['[unused2]'+i for i in new_inputs_plea]
    new_inputs_defendant = ['[unused3]'+i for i in new_inputs_defendant]
    # inputs_fact后面添加[SEP]特殊符号。需要考虑如果是被截断，则[SEP]被截断，需要把最后一个token换成[SEP]
    new_inputs_fact = ['[unused4]'+i+'[SEP]' for i in new_inputs_fact]

    # 处理 new_inputs_plaintiff
    if new_inputs_plaintiff:
        # 对new_inputs_plaintiff进行编码
        encoded_dict_plaintiff = tokenizer.batch_encode_plus(
                            new_inputs_plaintiff,                      # 输入的文本
                            add_special_tokens = False,   # 添加 [CLS] 和 [SEP]
                            max_length = args.plaintif_max_length, # 设置最大长度
                            padding='max_length', truncation=True,     # 对输入进行 padding 操作
                            return_attention_mask = True, # 生成 attention mask
                            return_tensors = 'pt'        # 返回 PyTorch 张量类型
                       )
        # 将编码后的文本添加到总输入中
        input_ids.append(encoded_dict_plaintiff['input_ids'])
        attention_masks.append(encoded_dict_plaintiff['attention_mask'])

    # 处理 new_inputs_plea
    if new_inputs_plea:
        # 对new_inputs_plea进行编码
        encoded_dict_plea = tokenizer.batch_encode_plus(
                            new_inputs_plea,
                            add_special_tokens = False,
                            max_length = args.plea_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_plea['input_ids'])
        attention_masks.append(encoded_dict_plea['attention_mask'])

    # 处理 new_inputs_defendant
    if new_inputs_defendant:
        # 对new_inputs_defendant进行编码
        encoded_dict_defendant = tokenizer.batch_encode_plus(
                            new_inputs_defendant,
                            add_special_tokens = False,
                            max_length = args.defendant_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_defendant['input_ids'])
        attention_masks.append(encoded_dict_defendant['attention_mask'])

    # 处理 new_inputs_fact
    if new_inputs_fact:
        # 对new_inputs_fact进行编码
        encoded_dict_fact = tokenizer.batch_encode_plus(
                            new_inputs_fact,
                            add_special_tokens = False,
                            max_length = args.fact_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_fact['input_ids'])
        attention_masks.append(encoded_dict_fact['attention_mask'])

    input_ids = torch.cat(input_ids, dim=1)
    attention_masks = torch.cat(attention_masks, dim=1)
    
    # 若tensor最后一列不为[PAD]的id（0），说明[SEP]被截断了，则需要把最后一列不为[PAD]的值替换为[SEP]的id（102）
    input_ids[:, -1] = torch.where(input_ids[:, -1] == tokenizer.pad_token_id, input_ids[:, -1], 102)
   
    # input_ids = [CLS] [unused1] inputs_plaintiff [unused2] 单条inputs_plea [unused3] inputs_defendant [unused4] inputs_fact [SEP]
    
    # 测试切片是否正确：
    # fjp_claim_ids = input_ids.narrow(1, 0, args.plaintif_max_length)
    # fjp_plea_ids = input_ids.narrow(1, args.plaintif_max_length, args.plea_max_length)
    # fjp_argument_ids = input_ids.narrow(1, args.plaintif_max_length + args.plea_max_length, args.defendant_max_length)
    # fjp_fact_ids = input_ids.narrow(1, args.plaintif_max_length + args.plea_max_length + args.defendant_max_length, args.fact_max_length)
    
    # print(f"fjp_claim_ids: {fjp_claim_ids}")
    # print(f"fjp_plea_ids: {fjp_plea_ids}")
    # print(f"fjp_argument_ids: {fjp_argument_ids}")
    # print(f"fjp_fact_ids: {fjp_fact_ids}")
    # exit(-1)
    return input_ids, attention_masks


def get_lawformerRI_fjp_inputs(args, tokenizer, inputs_plaintiff, inputs_plea, inputs_defendant, inputs_fact):
    '''
    任务三案由预测输入的截断和padding。
    lawformerRI: remove invalid context, including claim and defendant.
    输入格式：[CLS] [unused1] inputs_plaintiff [unused2] inputs_plea [unused3] inputs_defendant [unused4] inputs_fact [SEP]
    用于多任务lawformer、lawformer_CKE、lawformer_TCKE模型。
    参数：
        tokenizer: 模型使用的tokenizer
        inputs_plaintiff: 原告诉称内容
        inputs_plea:  原告诉求内容，可能包含一个或者多个，如果是多个，则需要拆分，因此batch_size会发生变化。
        inputs_defendant: 被告辩称内容
        inputs_fact: 事实描述的内容
    特殊符号表示：
        [unused1]：原告诉称
        [unused2]：原告诉请
        [unused3]：被告辩称
        [unused4]：事实描述
        [unused5]：案由标签描述
        [unused6]：通用法条描述
        [unused7]：特定法条描述
    '''
    # 初始化输入和掩码
    input_ids = []
    attention_masks = []
    
    # 最大长度断言
    assert args.input_max_length == args.plaintif_max_length + args.plea_max_length + args.defendant_max_length + args.fact_max_length
    
    # 根据诉求数量重组数据
    new_inputs_plaintiff = []
    new_inputs_plea = []
    new_inputs_defendant = []
    new_inputs_fact = []
    for plaintiff, pleas, defendant, fact in zip(inputs_plaintiff, inputs_plea, inputs_defendant, inputs_fact):
        # 单条样本可能包含多个诉求
        for plea in pleas:
            new_inputs_plaintiff.append(plaintiff)
            new_inputs_plea.append(plea)
            new_inputs_defendant.append(defendant)
            new_inputs_fact.append(fact)
    assert len(new_inputs_plaintiff) == len(new_inputs_plea)
    
    batch_size = len(new_inputs_plaintiff)
    # 依次为claim、plea、argument、fact权重
    Context_Weight = torch.ones(batch_size, 4).to(device=args.device)

    # new_inputs_plaintiff前面添加[CLS]特殊符号
    new_inputs_plaintiff = ['[CLS][unused1]'+i for i in new_inputs_plaintiff]
    new_inputs_plea = ['[unused2]'+i for i in new_inputs_plea]
    new_inputs_defendant = ['[unused3]'+i for i in new_inputs_defendant]
    # inputs_fact后面添加[SEP]特殊符号。需要考虑如果是被截断，则[SEP]被截断，需要把最后一个token换成[SEP]
    new_inputs_fact = ['[unused4]'+i+'[SEP]' for i in new_inputs_fact]

    # 处理 new_inputs_plaintiff
    if new_inputs_plaintiff:
        # 对new_inputs_plaintiff进行编码
        encoded_dict_plaintiff = tokenizer.batch_encode_plus(
                            new_inputs_plaintiff,                      # 输入的文本
                            add_special_tokens = False,   # 添加 [CLS] 和 [SEP]
                            max_length = args.plaintif_max_length, # 设置最大长度
                            padding='max_length', truncation=True,     # 对输入进行 padding 操作
                            return_attention_mask = True, # 生成 attention mask
                            return_tensors = 'pt'        # 返回 PyTorch 张量类型
                       )
        plaintiff_attention_mask = encoded_dict_plaintiff['attention_mask']
        # print(f"new_inputs_plaintiff: {new_inputs_plaintiff}")
        # print(f"处理前，plaintiff_attention_mask: {plaintiff_attention_mask[:,:3]}")

        # 如果原告诉称内容为空，则将其mask全置为0，将其对应权重置为0。
        for idx, text in enumerate(new_inputs_plaintiff):
            # 如果长度小于等于5，则可能为None等无效内容。
            if len(text.replace("[CLS][unused1]",""))<=5:
                plaintiff_attention_mask[idx, :] = 0
                Context_Weight[idx, 0] = 0
        
        # print(f"处理后，plaintiff_attention_mask: {plaintiff_attention_mask[:,:3]}")
        # print(f"Context_Weight: {Context_Weight}")
        # 将编码后的文本添加到总输入中
        input_ids.append(encoded_dict_plaintiff['input_ids'])
        attention_masks.append(plaintiff_attention_mask)

    # 处理 new_inputs_plea
    if new_inputs_plea:
        # 对new_inputs_plea进行编码
        encoded_dict_plea = tokenizer.batch_encode_plus(
                            new_inputs_plea,
                            add_special_tokens = False,
                            max_length = args.plea_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_plea['input_ids'])
        attention_masks.append(encoded_dict_plea['attention_mask'])

    # 处理 new_inputs_defendant
    if new_inputs_defendant:
        # 对new_inputs_defendant进行编码
        encoded_dict_defendant = tokenizer.batch_encode_plus(
                            new_inputs_defendant,
                            add_special_tokens = False,
                            max_length = args.defendant_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        defendant_attention_mask = encoded_dict_defendant['attention_mask']
        # print(f"new_inputs_defendant: {new_inputs_defendant}")
        # print(f"处理前defendant_attention_mask: {defendant_attention_mask}")

        # 如果被告诉称内容为空，则将其mask全置为0，将其对应权重置为0。
        for idx, text in enumerate(new_inputs_defendant):
            # 如果长度小于等于5，则可能为None等无效内容。
            if len(text.replace("[unused3]",""))<=5:
                defendant_attention_mask[idx, :] = 0
                Context_Weight[idx, 2] = 0

        # print(f"处理后defendant_attention_mask: {defendant_attention_mask[:,:3]}")
        # print(f"encoded_dict_defendant['input_ids']: {encoded_dict_defendant['input_ids'].shape}")
        # print(f"处理后defendant_attention_mask: {defendant_attention_mask.shape}")
        # print(f"Context_Weight: {Context_Weight}")
        input_ids.append(encoded_dict_defendant['input_ids'])
        attention_masks.append(defendant_attention_mask)

    # 处理 new_inputs_fact
    if new_inputs_fact:
        # 对new_inputs_fact进行编码
        encoded_dict_fact = tokenizer.batch_encode_plus(
                            new_inputs_fact,
                            add_special_tokens = False,
                            max_length = args.fact_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_fact['input_ids'])
        attention_masks.append(encoded_dict_fact['attention_mask'])

    input_ids = torch.cat(input_ids, dim=1)
    attention_masks = torch.cat(attention_masks, dim=1)
    
    # 若tensor最后一列不为[PAD]的id（0），说明[SEP]被截断了，则需要把最后一列不为[PAD]的值替换为[SEP]的id（102）
    input_ids[:, -1] = torch.where(input_ids[:, -1] == tokenizer.pad_token_id, input_ids[:, -1], 102)
   
    # input_ids = [CLS] [unused1] inputs_plaintiff [unused2] 单条inputs_plea [unused3] inputs_defendant [unused4] inputs_fact [SEP]
    return input_ids, attention_masks, Context_Weight/torch.sum(Context_Weight, dim=1).unsqueeze(-1)


def get_lawformerTAKE_fjp_inputs(args, tokenizer, inputs_plaintiff, inputs_plea, inputs_defendant, inputs_fact, gen_articles_context, gen_article_logits, spe_articles_context, spe_article_logits):
    '''
    任务三案由预测输入的截断和padding。
    输入格式：[CLS] [unused1] inputs_plaintiff [unused2] inputs_plea [unused3] inputs_defendant [unused4] inputs_fact [unused6] 通用Topk法条描述 [unused7] 特定Topk法条描述 [SEP]
    用于多任务lawformer_TCAKE模型。
    参数：
        tokenizer: 模型使用的tokenizer
        inputs_plaintiff: 原告诉称内容
        inputs_plea:  原告诉求内容，可能包含一个或者多个，如果是多个，则需要拆分，因此batch_size会发生变化。
        inputs_defendant: 被告辩称内容
        inputs_fact: 事实描述的内容
    特殊符号表示：
        [unused1]：原告诉称
        [unused2]：原告诉请
        [unused3]：被告辩称
        [unused4]：事实描述
        [unused5]：案由标签描述
        [unused6]：通用法条描述
        [unused7]：特定法条描述
    '''
    # 初始化输入和掩码
    input_ids = []
    attention_masks = []
    
    # 最大长度断言
    assert args.input_max_length == args.plaintif_max_length + args.plea_max_length + args.defendant_max_length + args.fact_max_length + args.general_article_max_length + args.specific_article_max_length

    # 取logits的top-k法条标签预测
    # 获取每行最大的两个值以及对应的位置
    gen_art_top_k_values, gen_art_top_k_indices = torch.topk(gen_article_logits, args.article_top_k, dim=1)

    spe_art_top_k_values, spe_art_top_k_indices = torch.topk(spe_article_logits, args.article_top_k, dim=1)
    
    inputs_topk_gen_art_context = []
    for top_k_indice in gen_art_top_k_indices:
        sample_gen_art_context = ['[unused6]']
        for indice in top_k_indice:
            sample_gen_art_context.append(gen_articles_context[indice])
        inputs_topk_gen_art_context.append(''.join(sample_gen_art_context))
    
    inputs_topk_spe_art_context = []
    for top_k_indice in spe_art_top_k_indices:
        sample_spe_art_context = ['[unused7]']
        for indice in top_k_indice:
            sample_spe_art_context.append(spe_articles_context[indice])
        sample_spe_art_context.append('[SEP]') # 由于是结尾，需要加上[SEP]
        inputs_topk_spe_art_context.append(''.join(sample_spe_art_context))
    

    # 根据诉求数量重组数据
    new_inputs_plaintiff = []
    new_inputs_plea = []
    new_inputs_defendant = []
    new_inputs_fact = []
    new_inputs_topk_gen_art_context = []
    new_inputs_topk_spe_art_context = []
    for plaintiff, pleas, defendant, fact, gen_art, spe_art in zip(inputs_plaintiff, inputs_plea, inputs_defendant, inputs_fact, inputs_topk_gen_art_context, inputs_topk_spe_art_context):
        # 单条样本可能包含多个诉求
        for plea in pleas:
            new_inputs_plaintiff.append(plaintiff)
            new_inputs_plea.append(plea)
            new_inputs_defendant.append(defendant)
            new_inputs_fact.append(fact)
            new_inputs_topk_gen_art_context.append(gen_art)
            new_inputs_topk_spe_art_context.append(spe_art)
    assert len(new_inputs_plaintiff) == len(new_inputs_plea)

    # new_inputs_plaintiff前面添加[CLS]特殊符号
    new_inputs_plaintiff = ['[CLS][unused1]'+i for i in new_inputs_plaintiff]
    new_inputs_plea = ['[unused2]'+i for i in new_inputs_plea]
    new_inputs_defendant = ['[unused3]'+i for i in new_inputs_defendant]
    new_inputs_fact = ['[unused4]'+i for i in new_inputs_fact]
    
    # 处理 new_inputs_plaintiff
    if new_inputs_plaintiff:
        # 对new_inputs_plaintiff进行编码
        encoded_dict_plaintiff = tokenizer.batch_encode_plus(
                            new_inputs_plaintiff,                      # 输入的文本
                            add_special_tokens = False,   # 添加 [CLS] 和 [SEP]
                            max_length = args.plaintif_max_length, # 设置最大长度
                            padding='max_length', truncation=True,     # 对输入进行 padding 操作
                            return_attention_mask = True, # 生成 attention mask
                            return_tensors = 'pt'        # 返回 PyTorch 张量类型
                       )
        # 将编码后的文本添加到总输入中
        input_ids.append(encoded_dict_plaintiff['input_ids'])
        attention_masks.append(encoded_dict_plaintiff['attention_mask'])
    
    # 处理 new_inputs_plea
    if new_inputs_plea:
        # 对new_inputs_plea进行编码
        encoded_dict_plea = tokenizer.batch_encode_plus(
                            new_inputs_plea,
                            add_special_tokens = False,
                            max_length = args.plea_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_plea['input_ids'])
        attention_masks.append(encoded_dict_plea['attention_mask'])

    # 处理 new_inputs_defendant
    if new_inputs_defendant:
        # 对new_inputs_defendant进行编码
        encoded_dict_defendant = tokenizer.batch_encode_plus(
                            new_inputs_defendant,
                            add_special_tokens = False,
                            max_length = args.defendant_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_defendant['input_ids'])
        attention_masks.append(encoded_dict_defendant['attention_mask'])

    # 处理 new_inputs_fact
    if new_inputs_fact:
        # 对new_inputs_fact进行编码
        encoded_dict_fact = tokenizer.batch_encode_plus(
                            new_inputs_fact,
                            add_special_tokens = False,
                            max_length = args.fact_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_fact['input_ids'])
        attention_masks.append(encoded_dict_fact['attention_mask'])


    # 处理 new_inputs_topk_gen_art_context
    if new_inputs_topk_gen_art_context:
        # 对new_inputs_topk_gen_art_context进行编码
        encoded_dict_topk_gen_art = tokenizer.batch_encode_plus(
                            new_inputs_topk_gen_art_context,
                            add_special_tokens = False,
                            max_length = args.general_article_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_topk_gen_art['input_ids'])
        attention_masks.append(encoded_dict_topk_gen_art['attention_mask'])

    # 处理 new_inputs_topk_spe_art_context
    if new_inputs_topk_spe_art_context:
        # 对new_inputs_topk_spe_art_context进行编码
        encoded_dict_topk_spe_art = tokenizer.batch_encode_plus(
                            new_inputs_topk_spe_art_context,
                            add_special_tokens = False,
                            max_length = args.specific_article_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_topk_spe_art['input_ids'])
        attention_masks.append(encoded_dict_topk_spe_art['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=1)
    attention_masks = torch.cat(attention_masks, dim=1)
    
    # 若tensor最后一列不为[PAD]的id（0），说明[SEP]被截断了，则需要把最后一列不为[PAD]的值替换为[SEP]的id（102）
    input_ids[:, -1] = torch.where(input_ids[:, -1] == tokenizer.pad_token_id, input_ids[:, -1], 102)
   
    # input_ids = [CLS] [unused1] inputs_plaintiff [unused2] 单条inputs_plea [unused3] inputs_defendant [unused4] inputs_fact [unused6] 通用Topk法条描述 [unused7] 特定Topk法条描述 [SEP]
    return input_ids, attention_masks


def get_lawformerTFKE_fjp_inputs(args, tokenizer, inputs_plaintiff, inputs_plea, inputs_defendant, inputs_fact, fjp_labels_context):
    '''
    任务三案由预测输入的截断和padding。
    输入格式：[CLS] [unused1] inputs_plaintiff [unused2] inputs_plea [unused3] inputs_defendant [unused4] inputs_fact [unused8] 最终判决标签描述 [SEP]
    用于多任务lawformer_TCAKE模型。
    参数：
        tokenizer: 模型使用的tokenizer
        inputs_plaintiff: 原告诉称内容
        inputs_plea:  原告诉求内容，可能包含一个或者多个，如果是多个，则需要拆分，因此batch_size会发生变化。
        inputs_defendant: 被告辩称内容
        inputs_fact: 事实描述的内容
        fjp_labels_context: 最终判决标签描述
    特殊符号表示：
        [unused1]：原告诉称
        [unused2]：原告诉请
        [unused3]：被告辩称
        [unused4]：事实描述
        [unused5]：案由标签描述
        [unused6]：通用法条描述
        [unused7]：特定法条描述
    '''
    # 初始化输入和掩码
    input_ids = []
    attention_masks = []
    
    # 最大长度断言
    assert args.input_max_length == args.plaintif_max_length + args.plea_max_length + args.defendant_max_length + args.fact_max_length + args.fjp_max_length

    # 最终判决标签描述
    inputs_fjp_context = ''.join(fjp_labels_context)

    # 根据诉求数量重组数据
    new_inputs_plaintiff = []
    new_inputs_plea = []
    new_inputs_defendant = []
    new_inputs_fact = []
    new_inputs_fjp_context = []
    for plaintiff, pleas, defendant, fact in zip(inputs_plaintiff, inputs_plea, inputs_defendant, inputs_fact):
        # 单条样本可能包含多个诉求
        for plea in pleas:
            new_inputs_plaintiff.append(plaintiff)
            new_inputs_plea.append(plea)
            new_inputs_defendant.append(defendant)
            new_inputs_fact.append(fact)
            # 不同的样本下fjp标签空间是一样的。
            new_inputs_fjp_context.append(inputs_fjp_context)
    assert len(new_inputs_plaintiff) == len(new_inputs_plea)

    # new_inputs_plaintiff前面添加[CLS]特殊符号
    new_inputs_plaintiff = ['[CLS][unused1]'+i for i in new_inputs_plaintiff]
    new_inputs_plea = ['[unused2]'+i for i in new_inputs_plea]
    new_inputs_defendant = ['[unused3]'+i for i in new_inputs_defendant]
    new_inputs_fact = ['[unused4]'+i for i in new_inputs_fact]
     # inputs_fjp_context长度固定，因此不会被截断
    new_inputs_fjp_context = ['[unused8]'+i+'[SEP]' for i in new_inputs_fjp_context]
    
    # 处理 new_inputs_plaintiff
    if new_inputs_plaintiff:
        # 对new_inputs_plaintiff进行编码
        encoded_dict_plaintiff = tokenizer.batch_encode_plus(
                            new_inputs_plaintiff,                      # 输入的文本
                            add_special_tokens = False,   # 添加 [CLS] 和 [SEP]
                            max_length = args.plaintif_max_length, # 设置最大长度
                            padding='max_length', truncation=True,     # 对输入进行 padding 操作
                            return_attention_mask = True, # 生成 attention mask
                            return_tensors = 'pt'        # 返回 PyTorch 张量类型
                       )
        # 将编码后的文本添加到总输入中
        input_ids.append(encoded_dict_plaintiff['input_ids'])
        attention_masks.append(encoded_dict_plaintiff['attention_mask'])
    
    # 处理 new_inputs_plea
    if new_inputs_plea:
        # 对new_inputs_plea进行编码
        encoded_dict_plea = tokenizer.batch_encode_plus(
                            new_inputs_plea,
                            add_special_tokens = False,
                            max_length = args.plea_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_plea['input_ids'])
        attention_masks.append(encoded_dict_plea['attention_mask'])

    # 处理 new_inputs_defendant
    if new_inputs_defendant:
        # 对new_inputs_defendant进行编码
        encoded_dict_defendant = tokenizer.batch_encode_plus(
                            new_inputs_defendant,
                            add_special_tokens = False,
                            max_length = args.defendant_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_defendant['input_ids'])
        attention_masks.append(encoded_dict_defendant['attention_mask'])

    # 处理 new_inputs_fact
    if new_inputs_fact:
        # 对new_inputs_fact进行编码
        encoded_dict_fact = tokenizer.batch_encode_plus(
                            new_inputs_fact,
                            add_special_tokens = False,
                            max_length = args.fact_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_fact['input_ids'])
        attention_masks.append(encoded_dict_fact['attention_mask'])
    
    # 处理 new_inputs_fjp_context
    if new_inputs_fjp_context:
        # 对new_inputs_fjp_context进行编码
        encoded_dict_fjp_context = tokenizer.batch_encode_plus(
                            new_inputs_fjp_context,
                            add_special_tokens = False,
                            max_length = args.fjp_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_fjp_context['input_ids'])
        attention_masks.append(encoded_dict_fjp_context['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=1)
    attention_masks = torch.cat(attention_masks, dim=1)
   
    # input_ids = [CLS] [unused1] inputs_plaintiff [unused2] 单条inputs_plea [unused3] inputs_defendant [unused4] inputs_fact [unused8] 最终判决标签描述 [SEP]
    return input_ids, attention_masks


def get_lawformer_fjp_MRC_inputs(args, tokenizer, inputs_plaintiff, inputs_plea, inputs_defendant, inputs_fact, fjp_label_name):
    '''
    任务三案由预测输入的截断和padding。
    输入格式：[CLS] [MASK] 最终判决标签描述1 [MASK] 最终判决标签描述2 [MASK] 最终判决标签描述3 [SEP] [unused2] inputs_plea [SEP] [unused1] inputs_plaintiff [unused3] inputs_defendant [unused4] inputs_fact [SEP]
    用于lawformer_MRC模型。
    参数：
        tokenizer: 模型使用的tokenizer
        inputs_plaintiff: 原告诉称内容
        inputs_plea:  原告诉求内容，可能包含一个或者多个，如果是多个，则需要拆分，因此batch_size会发生变化。
        inputs_defendant: 被告辩称内容
        inputs_fact: 事实描述的内容
        fjp_label_name: 最终判决标签描述
    特殊符号表示：
        [unused1]：原告诉称
        [unused2]：原告诉请
        [unused3]：被告辩称
        [unused4]：事实描述
        [unused5]：案由标签描述
        [unused6]：通用法条描述
        [unused7]：特定法条描述
    '''    
    # 最大长度断言
    assert args.input_max_length == args.plaintif_max_length + args.plea_max_length + args.defendant_max_length + args.fact_max_length + args.all_fjp_name_max_length

    # 最终判决标签描述
    # [MASK] fjp_label1 [MASK] fjp_label2 ...
    all_fjp_label_name = ''.join([tokenizer.mask_token + label for label in fjp_label_name])
    inputs_fjp_context = ''.join(all_fjp_label_name)

    # 根据诉求数量重组数据
    new_inputs_plaintiff = []
    new_inputs_plea = []
    new_inputs_defendant = []
    new_inputs_fact = []
    new_inputs_fjp_context = []
    for plaintiff, pleas, defendant, fact in zip(inputs_plaintiff, inputs_plea, inputs_defendant, inputs_fact):
        # 单条样本可能包含多个诉求
        for plea in pleas:
            new_inputs_plaintiff.append(plaintiff)
            new_inputs_plea.append(plea)
            new_inputs_defendant.append(defendant)
            new_inputs_fact.append(fact)
            # 不同的样本下fjp标签空间是一样的。
            new_inputs_fjp_context.append(inputs_fjp_context)
    assert len(new_inputs_plaintiff) == len(new_inputs_plea)
    
    # [SEP]用于分隔 option、question、passage内容
    # option内容：[CLS] [MASK] fjp_label1 [MASK] fjp_label2 [MASK] fjp_label3 [SEP]
    new_inputs_fjp_context = ['[CLS]'+i+'[SEP]' for i in new_inputs_fjp_context]
    
    # question内容: [unused2] plea内容 [SEP]
    new_inputs_plea = ['[unused2]'+i+'[SEP]' for i in new_inputs_plea]

    # passage内容: [unused1] 诉称内容 [unused3] 辩称内容 [unused4] 事实描述 [SEP]
    new_inputs_plaintiff = ['[unused1]'+i for i in new_inputs_plaintiff]
    new_inputs_defendant = ['[unused3]'+i for i in new_inputs_defendant]
    new_inputs_fact = ['[unused4]'+i+'[SEP]' for i in new_inputs_fact]

    # 处理option内容： new_inputs_fjp_context。长度固定，最大长度可以设置为恰好覆盖，因此[SEP]不会被截断。
    if new_inputs_fjp_context:
        # 对new_inputs_fjp_context进行编码
        encoded_dict_fjp_context = tokenizer.batch_encode_plus(
                            new_inputs_fjp_context,
                            add_special_tokens = False,
                            max_length = args.all_fjp_name_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        option_input_ids = encoded_dict_fjp_context['input_ids']
        option_attention_masks = encoded_dict_fjp_context['attention_mask']
        # print(f"option_input_ids: {option_input_ids[0]}")
        # print(f"option_attention_masks: {option_attention_masks[0]}")

    # 处理question内容：new_inputs_plea
    if new_inputs_plea:
        # 对new_inputs_plea进行编码
        encoded_dict_plea = tokenizer.batch_encode_plus(
                            new_inputs_plea,
                            add_special_tokens = False,
                            max_length = args.plea_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        question_input_ids = encoded_dict_plea['input_ids']
        question_attention_masks = encoded_dict_plea['attention_mask']
        # print(f"question_input_ids[0]:{list(question_input_ids[0].cpu().numpy())}")
        # print(f"question_attention_masks[0]:{question_attention_masks[0]}")

    # 若tensor最后一列不为[PAD]的id（0），说明[SEP]被截断了，则需要把最后一列不为[PAD]的值替换为[SEP]的id（102）
    question_input_ids[:, -1] = torch.where(question_input_ids[:, -1] == tokenizer.pad_token_id, question_input_ids[:, -1], 102)
  
    # 处理passage内容：new_inputs_plaintiff, new_inputs_defendant, new_inputs_fact
    passage_input_ids = []
    passage_attention_masks = []
    # 处理 new_inputs_plaintiff
    if new_inputs_plaintiff:
        # 对new_inputs_plaintiff进行编码
        encoded_dict_plaintiff = tokenizer.batch_encode_plus(
                            new_inputs_plaintiff,                      # 输入的文本
                            add_special_tokens = False,   # 添加 [CLS] 和 [SEP]
                            max_length = args.plaintif_max_length, # 设置最大长度
                            padding='max_length', truncation=True,     # 对输入进行 padding 操作
                            return_attention_mask = True, # 生成 attention mask
                            return_tensors = 'pt'        # 返回 PyTorch 张量类型
                       )
        # 将编码后的文本添加到总输入中
        passage_input_ids.append(encoded_dict_plaintiff['input_ids'])
        passage_attention_masks.append(encoded_dict_plaintiff['attention_mask'])

    # 处理 new_inputs_defendant
    if new_inputs_defendant:
        # 对new_inputs_defendant进行编码
        encoded_dict_defendant = tokenizer.batch_encode_plus(
                            new_inputs_defendant,
                            add_special_tokens = False,
                            max_length = args.defendant_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        passage_input_ids.append(encoded_dict_defendant['input_ids'])
        passage_attention_masks.append(encoded_dict_defendant['attention_mask'])

    # 处理 new_inputs_fact
    if new_inputs_fact:
        # 对new_inputs_fact进行编码
        encoded_dict_fact = tokenizer.batch_encode_plus(
                            new_inputs_fact,
                            add_special_tokens = False,
                            max_length = args.fact_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        passage_input_ids.append(encoded_dict_fact['input_ids'])
        passage_attention_masks.append(encoded_dict_fact['attention_mask'])

    
    # 拼接passage的内容
    passage_input_ids = torch.cat(passage_input_ids, dim=1)
    passage_attention_masks = torch.cat(passage_attention_masks, dim=1)

    # 若tensor最后一列不为[PAD]的id（0），说明[SEP]被截断了，则需要把最后一列不为[PAD]的值替换为[SEP]的id（102）
    passage_input_ids[:, -1] = torch.where(passage_input_ids[:, -1] == tokenizer.pad_token_id, passage_input_ids[:, -1], 102)
    # print(f"passage_input_ids[0]: {list(passage_input_ids[0].cpu().numpy())}")
    # print(f"passage_attention_masks[0]: {passage_attention_masks[0]}")

    # 按照顺序拼接option, question, passage.
    input_ids = torch.cat([option_input_ids, question_input_ids, passage_input_ids], dim=1)
    attention_masks = torch.cat([option_attention_masks, question_attention_masks, passage_attention_masks], dim=1)
 

    # print(f"input_ids[0]: {list(input_ids[0].cpu().numpy())}")
    # print(f"attention_masks[0]: {attention_masks[0]}")
    # input_ids = [CLS] [MASK] 最终判决标签描述1 [MASK] 最终判决标签描述2 [MASK] 最终判决标签描述3 [SEP] [unused2] inputs_plea [SEP] [unused1] inputs_plaintiff [unused3] inputs_defendant [unused4] inputs_fact [SEP] 
    return input_ids, attention_masks


def get_lawformerTAKE_fjp_MRC_inputs(args, tokenizer, inputs_plaintiff, inputs_plea, inputs_defendant, inputs_fact, fjp_label_name, gen_articles_context, gen_article_logits, spe_articles_context, spe_article_logits):
    '''
    任务三案由预测输入的截断和padding。
    输入格式：[CLS] [MASK] 最终判决标签描述1 [MASK] 最终判决标签描述2 [MASK] 最终判决标签描述3 [SEP] [unused2] inputs_plea [SEP] [unused1] inputs_plaintiff [unused3] inputs_defendant [unused4] inputs_fact [unused6] 通用Topk法条描述 [unused7] 特定Topk法条描述 [SEP]
    用于lawformer_MRC模型。
    参数：
        tokenizer: 模型使用的tokenizer
        inputs_plaintiff: 原告诉称内容
        inputs_plea:  原告诉求内容，可能包含一个或者多个，如果是多个，则需要拆分，因此batch_size会发生变化。
        inputs_defendant: 被告辩称内容
        inputs_fact: 事实描述的内容
        fjp_label_name: 最终判决标签描述
    特殊符号表示：
        [unused1]：原告诉称
        [unused2]：原告诉请
        [unused3]：被告辩称
        [unused4]：事实描述
        [unused5]：案由标签描述
        [unused6]：通用法条描述
        [unused7]：特定法条描述
    '''    
    # 最大长度断言
    assert args.input_max_length == args.plaintif_max_length + args.plea_max_length + args.defendant_max_length + args.fact_max_length + args.all_fjp_name_max_length + args.general_article_max_length + args.specific_article_max_length

    # 取logits的top-k法条标签预测
    # 获取每行最大的两个值以及对应的位置
    gen_art_top_k_values, gen_art_top_k_indices = torch.topk(gen_article_logits, args.article_top_k, dim=1)

    spe_art_top_k_values, spe_art_top_k_indices = torch.topk(spe_article_logits, args.article_top_k, dim=1)
    
    inputs_topk_gen_art_context = []
    for top_k_indice in gen_art_top_k_indices:
        sample_gen_art_context = ['[unused6]']
        for indice in top_k_indice:
            sample_gen_art_context.append(gen_articles_context[indice])
        inputs_topk_gen_art_context.append(''.join(sample_gen_art_context))
    
    inputs_topk_spe_art_context = []
    for top_k_indice in spe_art_top_k_indices:
        sample_spe_art_context = ['[unused7]']
        for indice in top_k_indice:
            sample_spe_art_context.append(spe_articles_context[indice])
        sample_spe_art_context.append('[SEP]') # 由于是结尾，需要加上[SEP]
        inputs_topk_spe_art_context.append(''.join(sample_spe_art_context))
        
    # 最终判决标签描述
    # [MASK] fjp_label1 [MASK] fjp_label2 ...
    all_fjp_label_name = ''.join([tokenizer.mask_token + label for label in fjp_label_name])
    inputs_fjp_context = ''.join(all_fjp_label_name)

    # 根据诉求数量重组数据
    new_inputs_plaintiff = []
    new_inputs_plea = []
    new_inputs_defendant = []
    new_inputs_fact = []
    new_inputs_fjp_context = []
    new_inputs_topk_gen_art_context = []
    new_inputs_topk_spe_art_context = []

    for plaintiff, pleas, defendant, fact, gen_art, spe_art in zip(inputs_plaintiff, inputs_plea, inputs_defendant, inputs_fact, inputs_topk_gen_art_context, inputs_topk_spe_art_context):
        # 单条样本可能包含多个诉求
        for plea in pleas:
            new_inputs_plaintiff.append(plaintiff)
            new_inputs_plea.append(plea)
            new_inputs_defendant.append(defendant)
            new_inputs_fact.append(fact)
            # 不同的样本下fjp标签空间是一样的。
            new_inputs_fjp_context.append(inputs_fjp_context)
            new_inputs_topk_gen_art_context.append(gen_art)
            new_inputs_topk_spe_art_context.append(spe_art)
    assert len(new_inputs_plaintiff) == len(new_inputs_plea)
    
    # [SEP]用于分隔 option、question、passage内容
    # option内容：[CLS] [MASK] fjp_label1 [MASK] fjp_label2 [MASK] fjp_label3 [SEP]
    new_inputs_fjp_context = ['[CLS]'+i+'[SEP]' for i in new_inputs_fjp_context]
    
    # question内容: [unused2] plea内容 [SEP]
    new_inputs_plea = ['[unused2]'+i+'[SEP]' for i in new_inputs_plea]
    # passage内容: [unused1] 诉称内容 [unused3] 辩称内容 [unused4] 事实描述 [SEP]
    new_inputs_plaintiff = ['[unused1]'+i for i in new_inputs_plaintiff]
    new_inputs_defendant = ['[unused3]'+i for i in new_inputs_defendant]
    new_inputs_fact = ['[unused4]'+i for i in new_inputs_fact]

    # 1. 处理option内容： new_inputs_fjp_context。长度固定，最大长度可以设置为恰好覆盖，因此[SEP]不会被截断。
    if new_inputs_fjp_context:
        # 对new_inputs_fjp_context进行编码
        encoded_dict_fjp_context = tokenizer.batch_encode_plus(
                            new_inputs_fjp_context,
                            add_special_tokens = False,
                            max_length = args.all_fjp_name_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        option_input_ids = encoded_dict_fjp_context['input_ids']
        option_attention_masks = encoded_dict_fjp_context['attention_mask']
        # print(f"option_input_ids: {option_input_ids[0]}")
        # print(f"option_attention_masks: {option_attention_masks[0]}")

    # 2. 处理question内容：new_inputs_plea
    if new_inputs_plea:
        # 对new_inputs_plea进行编码
        encoded_dict_plea = tokenizer.batch_encode_plus(
                            new_inputs_plea,
                            add_special_tokens = False,
                            max_length = args.plea_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        question_input_ids = encoded_dict_plea['input_ids']
        question_attention_masks = encoded_dict_plea['attention_mask']
        # print(f"question_input_ids[0]:{list(question_input_ids[0].cpu().numpy())}")
        # print(f"question_attention_masks[0]:{question_attention_masks[0]}")

    # 若tensor最后一列不为[PAD]的id（0），说明[SEP]被截断了，则需要把最后一列不为[PAD]的值替换为[SEP]的id（102）
    question_input_ids[:, -1] = torch.where(question_input_ids[:, -1] == tokenizer.pad_token_id, question_input_ids[:, -1], 102)
  
    # 3. 处理passage内容：new_inputs_plaintiff, new_inputs_defendant, new_inputs_fact
    passage_input_ids = []
    passage_attention_masks = []
    # 处理 new_inputs_plaintiff
    if new_inputs_plaintiff:
        # 对new_inputs_plaintiff进行编码
        encoded_dict_plaintiff = tokenizer.batch_encode_plus(
                            new_inputs_plaintiff,                      # 输入的文本
                            add_special_tokens = False,   # 添加 [CLS] 和 [SEP]
                            max_length = args.plaintif_max_length, # 设置最大长度
                            padding='max_length', truncation=True,     # 对输入进行 padding 操作
                            return_attention_mask = True, # 生成 attention mask
                            return_tensors = 'pt'        # 返回 PyTorch 张量类型
                       )
        # 将编码后的文本添加到总输入中
        passage_input_ids.append(encoded_dict_plaintiff['input_ids'])
        passage_attention_masks.append(encoded_dict_plaintiff['attention_mask'])

    # 处理 new_inputs_defendant
    if new_inputs_defendant:
        # 对new_inputs_defendant进行编码
        encoded_dict_defendant = tokenizer.batch_encode_plus(
                            new_inputs_defendant,
                            add_special_tokens = False,
                            max_length = args.defendant_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        passage_input_ids.append(encoded_dict_defendant['input_ids'])
        passage_attention_masks.append(encoded_dict_defendant['attention_mask'])

    # 处理 new_inputs_fact
    if new_inputs_fact:
        # 对new_inputs_fact进行编码
        encoded_dict_fact = tokenizer.batch_encode_plus(
                            new_inputs_fact,
                            add_special_tokens = False,
                            max_length = args.fact_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        passage_input_ids.append(encoded_dict_fact['input_ids'])
        passage_attention_masks.append(encoded_dict_fact['attention_mask'])

    
    # 处理 new_inputs_topk_gen_art_context
    if new_inputs_topk_gen_art_context:
        # 对new_inputs_topk_gen_art_context进行编码
        encoded_dict_topk_gen_art = tokenizer.batch_encode_plus(
                            new_inputs_topk_gen_art_context,
                            add_special_tokens = False,
                            max_length = args.general_article_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        passage_input_ids.append(encoded_dict_topk_gen_art['input_ids'])
        passage_attention_masks.append(encoded_dict_topk_gen_art['attention_mask'])

    # 处理 new_inputs_topk_spe_art_context
    if new_inputs_topk_spe_art_context:
        # 对new_inputs_topk_spe_art_context进行编码
        encoded_dict_topk_spe_art = tokenizer.batch_encode_plus(
                            new_inputs_topk_spe_art_context,
                            add_special_tokens = False,
                            max_length = args.specific_article_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        passage_input_ids.append(encoded_dict_topk_spe_art['input_ids'])
        passage_attention_masks.append(encoded_dict_topk_spe_art['attention_mask'])

    # 拼接passage的内容
    passage_input_ids = torch.cat(passage_input_ids, dim=1)
    passage_attention_masks = torch.cat(passage_attention_masks, dim=1)

    # 若tensor最后一列不为[PAD]的id（0），说明[SEP]被截断了，则需要把最后一列不为[PAD]的值替换为[SEP]的id（102）
    passage_input_ids[:, -1] = torch.where(passage_input_ids[:, -1] == tokenizer.pad_token_id, passage_input_ids[:, -1], 102)
    # print(f"passage_input_ids[0]: {list(passage_input_ids[0].cpu().numpy())}")
    # print(f"passage_attention_masks[0]: {passage_attention_masks[0]}")

    # 按照顺序拼接option, question, passage.
    input_ids = torch.cat([option_input_ids, question_input_ids, passage_input_ids], dim=1)
    attention_masks = torch.cat([option_attention_masks, question_attention_masks, passage_attention_masks], dim=1)
 
    # input_ids = [CLS] [MASK] 最终判决标签描述1 [MASK] 最终判决标签描述2 [MASK] 最终判决标签描述3 [SEP] [unused2] inputs_plea [SEP] [unused1] inputs_plaintiff [unused3] inputs_defendant [unused4] inputs_fact [unused6] 通用Topk法条描述 [unused7] 特定Topk法条描述 [SEP] 
    return input_ids, attention_masks


def get_lawformerTAFKE_fjp_inputs(args, tokenizer, inputs_plaintiff, inputs_plea, inputs_defendant, inputs_fact, gen_articles_context, gen_article_logits, spe_articles_context, spe_article_logits, fjp_labels_context):
    '''
    任务三案由预测输入的截断和padding。
    输入格式：[CLS] [unused1] inputs_plaintiff [unused2] inputs_plea [unused3] inputs_defendant [unused4] inputs_fact [unused6] 通用Topk法条描述 [unused7] 特定Topk法条描述 [unused8] 最终判决标签描述 [SEP]
    用于多任务lawformer_TCAKE模型。
    参数：
        tokenizer: 模型使用的tokenizer
        inputs_plaintiff: 原告诉称内容
        inputs_plea:  原告诉求内容，可能包含一个或者多个，如果是多个，则需要拆分，因此batch_size会发生变化。
        inputs_defendant: 被告辩称内容
        inputs_fact: 事实描述的内容
        fjp_labels_context: 最终判决标签描述
    特殊符号表示：
        [unused1]：原告诉称
        [unused2]：原告诉请
        [unused3]：被告辩称
        [unused4]：事实描述
        [unused5]：案由标签描述
        [unused6]：通用法条描述
        [unused7]：特定法条描述
    '''
    # 初始化输入和掩码
    input_ids = []
    attention_masks = []
    
    # 最大长度断言
    assert args.input_max_length == args.plaintif_max_length + args.plea_max_length + args.defendant_max_length + args.fact_max_length + args.general_article_max_length + args.specific_article_max_length + args.fjp_max_length

    # 取logits的top-k法条标签预测
    # 获取每行最大的两个值以及对应的位置
    gen_art_top_k_values, gen_art_top_k_indices = torch.topk(gen_article_logits, args.article_top_k, dim=1)

    spe_art_top_k_values, spe_art_top_k_indices = torch.topk(spe_article_logits, args.article_top_k, dim=1)
    
    inputs_topk_gen_art_context = []
    for top_k_indice in gen_art_top_k_indices:
        sample_gen_art_context = ['[unused6]']
        for indice in top_k_indice:
            sample_gen_art_context.append(gen_articles_context[indice])
        inputs_topk_gen_art_context.append(''.join(sample_gen_art_context))
    
    inputs_topk_spe_art_context = []
    for top_k_indice in spe_art_top_k_indices:
        sample_spe_art_context = ['[unused7]']
        for indice in top_k_indice:
            sample_spe_art_context.append(spe_articles_context[indice])
        inputs_topk_spe_art_context.append(''.join(sample_spe_art_context))
    
    # 最终判决标签描述
    inputs_fjp_context = ''.join(fjp_labels_context)

    # 根据诉求数量重组数据
    new_inputs_plaintiff = []
    new_inputs_plea = []
    new_inputs_defendant = []
    new_inputs_fact = []
    new_inputs_topk_gen_art_context = []
    new_inputs_topk_spe_art_context = []
    new_inputs_fjp_context = []
    for plaintiff, pleas, defendant, fact, gen_art, spe_art in zip(inputs_plaintiff, inputs_plea, inputs_defendant, inputs_fact, inputs_topk_gen_art_context, inputs_topk_spe_art_context):
        # 单条样本可能包含多个诉求
        for plea in pleas:
            new_inputs_plaintiff.append(plaintiff)
            new_inputs_plea.append(plea)
            new_inputs_defendant.append(defendant)
            new_inputs_fact.append(fact)
            new_inputs_topk_gen_art_context.append(gen_art)
            new_inputs_topk_spe_art_context.append(spe_art)
            # 不同的样本下fjp标签空间是一样的。
            new_inputs_fjp_context.append(inputs_fjp_context)
    assert len(new_inputs_plaintiff) == len(new_inputs_plea)

    # new_inputs_plaintiff前面添加[CLS]特殊符号
    new_inputs_plaintiff = ['[CLS][unused1]'+i for i in new_inputs_plaintiff]
    new_inputs_plea = ['[unused2]'+i for i in new_inputs_plea]
    new_inputs_defendant = ['[unused3]'+i for i in new_inputs_defendant]
    new_inputs_fact = ['[unused4]'+i for i in new_inputs_fact]
    new_inputs_topk_gen_art_context = ['[unused6]'+i for i in new_inputs_topk_gen_art_context]
    new_inputs_topk_spe_art_context = ['[unused7]'+i for i in new_inputs_topk_spe_art_context]
     # inputs_fjp_context长度固定，因此不会被截断
    new_inputs_fjp_context = ['[unused8]'+i+'[SEP]' for i in new_inputs_fjp_context]
    
    # 处理 new_inputs_plaintiff
    if new_inputs_plaintiff:
        # 对new_inputs_plaintiff进行编码
        encoded_dict_plaintiff = tokenizer.batch_encode_plus(
                            new_inputs_plaintiff,                      # 输入的文本
                            add_special_tokens = False,   # 添加 [CLS] 和 [SEP]
                            max_length = args.plaintif_max_length, # 设置最大长度
                            padding='max_length', truncation=True,     # 对输入进行 padding 操作
                            return_attention_mask = True, # 生成 attention mask
                            return_tensors = 'pt'        # 返回 PyTorch 张量类型
                       )
        # 将编码后的文本添加到总输入中
        input_ids.append(encoded_dict_plaintiff['input_ids'])
        attention_masks.append(encoded_dict_plaintiff['attention_mask'])
    
    # 处理 new_inputs_plea
    if new_inputs_plea:
        # 对new_inputs_plea进行编码
        encoded_dict_plea = tokenizer.batch_encode_plus(
                            new_inputs_plea,
                            add_special_tokens = False,
                            max_length = args.plea_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_plea['input_ids'])
        attention_masks.append(encoded_dict_plea['attention_mask'])

    # 处理 new_inputs_defendant
    if new_inputs_defendant:
        # 对new_inputs_defendant进行编码
        encoded_dict_defendant = tokenizer.batch_encode_plus(
                            new_inputs_defendant,
                            add_special_tokens = False,
                            max_length = args.defendant_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_defendant['input_ids'])
        attention_masks.append(encoded_dict_defendant['attention_mask'])

    # 处理 new_inputs_fact
    if new_inputs_fact:
        # 对new_inputs_fact进行编码
        encoded_dict_fact = tokenizer.batch_encode_plus(
                            new_inputs_fact,
                            add_special_tokens = False,
                            max_length = args.fact_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_fact['input_ids'])
        attention_masks.append(encoded_dict_fact['attention_mask'])


    # 处理 new_inputs_topk_gen_art_context
    if new_inputs_topk_gen_art_context:
        # 对new_inputs_topk_gen_art_context进行编码
        encoded_dict_topk_gen_art = tokenizer.batch_encode_plus(
                            new_inputs_topk_gen_art_context,
                            add_special_tokens = False,
                            max_length = args.general_article_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_topk_gen_art['input_ids'])
        attention_masks.append(encoded_dict_topk_gen_art['attention_mask'])

    # 处理 new_inputs_topk_spe_art_context
    if new_inputs_topk_spe_art_context:
        # 对new_inputs_topk_spe_art_context进行编码
        encoded_dict_topk_spe_art = tokenizer.batch_encode_plus(
                            new_inputs_topk_spe_art_context,
                            add_special_tokens = False,
                            max_length = args.specific_article_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_topk_spe_art['input_ids'])
        attention_masks.append(encoded_dict_topk_spe_art['attention_mask'])
    
    # 处理 new_inputs_fjp_context
    if new_inputs_fjp_context:
        # 对new_inputs_fjp_context进行编码
        encoded_dict_fjp_context = tokenizer.batch_encode_plus(
                            new_inputs_fjp_context,
                            add_special_tokens = False,
                            max_length = args.fjp_max_length,
                            padding='max_length', truncation=True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        input_ids.append(encoded_dict_fjp_context['input_ids'])
        attention_masks.append(encoded_dict_fjp_context['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=1)
    attention_masks = torch.cat(attention_masks, dim=1)
   
    # input_ids = [CLS] [unused1] inputs_plaintiff [unused2] 单条inputs_plea [unused3] inputs_defendant [unused4] inputs_fact [unused6] 通用Topk法条描述 [unused7] 特定Topk法条描述 [unused8] 最终判决标签描述 [SEP]
    return input_ids, attention_masks


def labels_to_multihot(labels, num_classes=146):
    multihot_labels = torch.zeros(len(labels), num_classes)
    for i, label in enumerate(labels):
        for l in label:
            multihot_labels[i][l] = 1
    return multihot_labels


def get_precision_recall_f1(y_true: np.array, y_pred: np.array, average='micro'):
    precision = metrics.precision_score(
        y_true, y_pred, average=average, zero_division=0)
    recall = metrics.recall_score(
        y_true, y_pred, average=average, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average=average, zero_division=0)
    # 如果用这个计算multi-label设置，则是在计算绝对匹配率：对于每个样本来说，只有预测值与真实值完全相同的情况下才算预测正确。
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy, precision, recall, f1


def evaluate_civilJP(test_dataloader, model, device, args):
    model.eval()
    all_cause_pred_labels = []
    all_cause_gt_labels = []
    all_gen_articles_pred_labels = []
    all_gen_articles_gt_labels = []
    all_spe_articles_pred_labels = []
    all_spe_articles_gt_labels = []
    all_fjp_pred_labels = []
    all_fjp_gt_labels = []
    all_idx = []
    for i, data in enumerate(test_dataloader):    
        if i % 200 == 0:
            print(f'Processing samples {i * args.batch_size}')   
        # idx: 原样本序号 
        idx, plaintiff_text, plea_text, defendant_text, fact_text, cause_label_id, gen_article_label_id, spe_article_label_id, fjp_labels_id = data

        # move data to device
        cause_label_id = torch.from_numpy(np.array(cause_label_id)).to(device)
        gen_article_label_id = labels_to_multihot(gen_article_label_id, num_classes=args.gen_article_num_classes)
        gen_article_label_id = torch.from_numpy(np.array(gen_article_label_id)).to(device)
        
        spe_article_label_id = labels_to_multihot(spe_article_label_id, num_classes=args.spe_article_num_classes)
        spe_article_label_id = torch.from_numpy(np.array(spe_article_label_id)).to(device)
        
        # 包含多个诉求的情况，转化成多条样本。
        new_fjp_labels_id = []
        for fjps in fjp_labels_id:
            for fjp in fjps:
                new_fjp_labels_id.append(fjp)
        
        new_fjp_labels_id = torch.from_numpy(np.array(new_fjp_labels_id)).to(device)


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

    return cause_accuracy, cause_p_macro, cause_r_macro, cause_f1_macro, cause_p_micro, cause_r_micro, cause_f1_micro, gen_article_accuracy, gen_article_p_macro, gen_article_r_macro, gen_article_f1_macro, gen_article_p_micro, gen_article_r_micro, gen_article_f1_micro, spe_article_accuracy, spe_article_p_macro, spe_article_r_macro, spe_article_f1_macro, spe_article_p_micro, spe_article_r_micro, spe_article_f1_micro, fjp_accuracy, fjp_p_macro, fjp_r_macro, fjp_f1_macro, fjp_p_micro, fjp_r_micro, fjp_f1_micro


def evaluate_ST_civilJP(test_dataloader, model, device, args):
    '''single_task of civilJP'''
    model.eval()

    if args.model_variants in ["SingleTaskLawformerCivilJP_CCP", "SingleTaskT5CivilJP_CCP"]:
        all_cause_pred_labels = []
        all_cause_gt_labels = []

        for i, data in enumerate(test_dataloader):       
            # idx: 原样本序号 
            idx, plaintiff_text, plea_text, defendant_text, fact_text, cause_label_id, gen_article_label_id, spe_article_label_id, fjp_labels_id = data

            # move data to device
            cause_label_id = torch.from_numpy(np.array(cause_label_id)).to(device)

            # forward
            with torch.no_grad():
                cause_logits = model(plaintiff_text, plea_text, defendant_text, fact_text)

            # cause label prediction
            cause_pred_probs = cause_logits.softmax(dim=1).detach().cpu().numpy()
            cause_pred_label = np.argmax(cause_pred_probs, axis=1)
            all_cause_pred_labels.append(torch.from_numpy(cause_pred_label))
            all_cause_gt_labels.append(cause_label_id)

        # merge all label prediction
        all_cause_pred_labels = torch.cat(all_cause_pred_labels, dim=0).cpu().numpy()
        all_cause_gt_labels = torch.cat(all_cause_gt_labels, dim=0).cpu().numpy()
        print(f"Test dataset of cause number is:{len(all_cause_gt_labels)}")
        
        # cause prediction
        cause_accuracy, cause_p_macro, cause_r_macro, cause_f1_macro = get_precision_recall_f1(y_true=all_cause_gt_labels, y_pred=all_cause_pred_labels, average='macro')
        cause_accuracy, cause_p_micro, cause_r_micro, cause_f1_micro = get_precision_recall_f1(y_true=all_cause_gt_labels, y_pred=all_cause_pred_labels, average='micro')

        # write each class metric to csv
        if args.is_metric2csv == True:
            multiClass_metric2csv(all_cause_gt_labels, all_cause_pred_labels, args.cause_metric_save_path)
        
        return cause_accuracy, cause_p_macro, cause_r_macro, cause_f1_macro, cause_p_micro, cause_r_micro, cause_f1_micro
    elif args.model_variants in ["SingleTaskLawformerCivilJP_CLAP", "SingleTaskT5CivilJP_CLAP"]:
        all_gen_articles_pred_labels = []
        all_gen_articles_gt_labels = []
        all_spe_articles_pred_labels = []
        all_spe_articles_gt_labels = []
        for i, data in enumerate(test_dataloader):       
            # idx: 原样本序号 
            idx, plaintiff_text, plea_text, defendant_text, fact_text, cause_label_id, gen_article_label_id, spe_article_label_id, fjp_labels_id = data

            # move data to device
            gen_article_label_id = labels_to_multihot(gen_article_label_id, num_classes=args.gen_article_num_classes)
            gen_article_label_id = torch.from_numpy(np.array(gen_article_label_id)).to(device)
            
            spe_article_label_id = labels_to_multihot(spe_article_label_id, num_classes=args.spe_article_num_classes)
            spe_article_label_id = torch.from_numpy(np.array(spe_article_label_id)).to(device)
            
            # forward
            with torch.no_grad():
                gen_article_logits, spe_article_logits = model(plaintiff_text, plea_text, defendant_text, fact_text)

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

        all_gen_articles_pred_labels = torch.cat(all_gen_articles_pred_labels, dim=0).cpu().numpy()
        all_gen_articles_gt_labels = torch.cat(all_gen_articles_gt_labels, dim=0).cpu().numpy()

        all_spe_articles_pred_labels = torch.cat(all_spe_articles_pred_labels, dim=0).cpu().numpy()
        all_spe_articles_gt_labels = torch.cat(all_spe_articles_gt_labels, dim=0).cpu().numpy()
        print(f"Test dataset of gen article number is:{len(all_gen_articles_gt_labels)}")
        print(f"Test dataset of spe cause number is:{len(all_spe_articles_gt_labels)}")
  
        # general articles prediction
        gen_article_accuracy, gen_article_p_macro, gen_article_r_macro, gen_article_f1_macro = get_precision_recall_f1(y_true=all_gen_articles_gt_labels, y_pred=all_gen_articles_pred_labels, average='macro')
        gen_article_accuracy, gen_article_p_micro, gen_article_r_micro, gen_article_f1_micro = get_precision_recall_f1(y_true=all_gen_articles_gt_labels, y_pred=all_gen_articles_pred_labels, average='micro')

        # specific articles prediction
        spe_article_accuracy, spe_article_p_macro, spe_article_r_macro, spe_article_f1_macro = get_precision_recall_f1(y_true=all_spe_articles_gt_labels, y_pred=all_spe_articles_pred_labels, average='macro')
        spe_article_accuracy, spe_article_p_micro, spe_article_r_micro, spe_article_f1_micro = get_precision_recall_f1(y_true=all_spe_articles_gt_labels, y_pred=all_spe_articles_pred_labels, average='micro')

        # write each class metric to csv
        if args.is_metric2csv == True:
            multiLabel_metric2csv(all_gen_articles_gt_labels, all_gen_articles_pred_labels, args.gen_articles_metric_save_path)
            multiLabel_metric2csv(all_spe_articles_gt_labels, all_spe_articles_pred_labels, args.spe_articles_metric_save_path)

        return gen_article_accuracy, gen_article_p_macro, gen_article_r_macro, gen_article_f1_macro, gen_article_p_micro, gen_article_r_micro, gen_article_f1_micro, spe_article_accuracy, spe_article_p_macro, spe_article_r_macro, spe_article_f1_macro, spe_article_p_micro, spe_article_r_micro, spe_article_f1_micro
    elif args.model_variants in ["SingleTaskLawformerCivilJP_FJP", "SingleTaskLawformerCivilJP_FJP_v1", "SingleTaskLawformerCivilJP_FJP_v2", "SingleTaskLawformerCivilJP_FJP_v3", "SingleTaskLawformerCivilJP_FJP_v4", "SingleTaskT5CivilJP_FJP_add", "SingleTaskT5CivilJP_FJP_concat", "SingleTaskLawformerCivilJP_FJP_v5", "SingleTaskLawformerCivilJP_FJP_v6", "SingleTaskLawformerCivilJP_FJP_v7", "SingleTaskLawformerCivilJP_FJP_MRC"]:
        all_fjp_pred_labels = []
        all_fjp_gt_labels = []
        
        for i, data in enumerate(test_dataloader):       
            # idx: 原样本序号 
            idx, plaintiff_text, plea_text, defendant_text, fact_text, cause_label_id, gen_article_label_id, spe_article_label_id, fjp_labels_id = data

            # 包含多个诉求的情况，转化成多条样本。
            new_fjp_labels_id = []
            for fjps in fjp_labels_id:
                for fjp in fjps:
                    new_fjp_labels_id.append(fjp)
            
            new_fjp_labels_id = torch.from_numpy(np.array(new_fjp_labels_id)).to(device)

            # forward
            with torch.no_grad():
                fjp_logits = model(plaintiff_text, plea_text, defendant_text, fact_text)

            # fjp label prediction
            fjp_predictions = fjp_logits.softmax(dim=1).detach().cpu().numpy()
            fjp_pred_label = np.argmax(fjp_predictions, axis=1)
            all_fjp_pred_labels.append(torch.from_numpy(fjp_pred_label))
            all_fjp_gt_labels.append(new_fjp_labels_id)

        all_fjp_pred_labels = torch.cat(all_fjp_pred_labels, dim=0).cpu().numpy()
        all_fjp_gt_labels = torch.cat(all_fjp_gt_labels, dim=0).cpu().numpy()
        
        assert len(all_fjp_pred_labels) == len(all_fjp_gt_labels)
        
        print(f"Test dataset of fjp number is:{len(all_fjp_gt_labels)}")

        # fjp prediction
        fjp_accuracy, fjp_p_macro, fjp_r_macro, fjp_f1_macro = get_precision_recall_f1(y_true=all_fjp_gt_labels, y_pred=all_fjp_pred_labels, average='macro')
        fjp_accuracy, fjp_p_micro, fjp_r_micro, fjp_f1_micro = get_precision_recall_f1(y_true=all_fjp_gt_labels, y_pred=all_fjp_pred_labels, average='micro')

        # write each class metric to csv
        if args.is_metric2csv == True:
            multiClass_metric2csv(all_fjp_gt_labels, all_fjp_pred_labels, args.fjp_metric_save_path)

        return fjp_accuracy, fjp_p_macro, fjp_r_macro, fjp_f1_macro, fjp_p_micro, fjp_r_micro, fjp_f1_micro
    else:
        raise NameError