import torch
import torch.nn as nn
from transformers import BertModel, AutoModel, AutoTokenizer
import torch.nn.functional as F
from utils.CivilJP_utils import *
import logging
import codecs
import json

class KnowAugLawformerCivilJP(nn.Module):
    '''
    Knowledge-enhanced PLMs for civil-jurisdiction prediction.
    PLMs：lawformer.
 
    Model Name Lists:
    LawformerCivilJP_baseCLS: baseline
    LawformerCivilJP_baseMean: 未考虑任务依赖关系、结构感知、多视角特征的TML消融模型
    LawformerCivilJP: 多任务统一建模CJP三个任务
    KnowAugLawformerCivilJP_TD (TML)：引入任务依赖
    '''
    def __init__(self, args):
        super(KnowAugLawformerCivilJP, self).__init__()
        self.model = AutoModel.from_pretrained(args.PTM_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.PTM_path)
        # add "[unused*]" as special_tokens for idenity different texts
        #     特殊符号表示：
        #     [unused1]：原告诉称
        #     [unused2]：原告诉请
        #     [unused3]：被告辩称
        #     [unused4]：事实描述
        #     [unused5]：案由标签描述
        #     [unused6]：通用法条描述
        #     [unused7]：特定法条描述
        for i in ['[unused'+ str(i+1) + ']' for i in range(7)]:
            self.tokenizer.add_tokens([i], special_tokens=True)
        if args.is_train:
            # 分布式训练时
            self.device = args.local_rank
        else:
            # 验证时
            self.device = args.device 
            
        self.args = args
        self.cause_num_classes = args.cause_num_classes
        self.gen_article_num_classes = args.gen_article_num_classes
        self.spe_article_num_classes = args.spe_article_num_classes
        self.fjp_num_classes = args.fjp_num_classes

        self.CE_loss = nn.CrossEntropyLoss()
        self.BCE_loss = nn.BCELoss()
         
        if args.model_variants in ["LawformerCivilJP", "LawformerCivilJP_baseCLS", "LawformerCivilJP_baseMean"]:
            self.cause_cls_linear = nn.Linear(in_features=768, out_features=args.cause_num_classes)
            self.gen_article_cls_linear = nn.Linear(in_features=768, out_features=args.gen_article_num_classes)
            self.spe_article_cls_linear = nn.Linear(in_features=768, out_features=args.spe_article_num_classes)
            self.fjp_cls_linear = nn.Linear(in_features=768, out_features=args.fjp_num_classes)
        elif args.model_variants in ["KnowAugLawformerCivilJP_TD"]:
            self.cause_cls_linear = nn.Linear(in_features=3*768, out_features=args.cause_num_classes)
            self.gen_article_cls_linear = nn.Linear(in_features=2*768, out_features=args.gen_article_num_classes)
            self.spe_article_cls_linear = nn.Linear(in_features=2*768, out_features=args.spe_article_num_classes)
            self.fjp_cls_linear = nn.Linear(in_features=6*768, out_features=args.fjp_num_classes)
            
            # 载入案由标签信息: 按照标签位置依次排列。
            self.cause_labels_context = [i.rstrip('\r\n').split('\t')[1] for i in codecs.open(args.cause_context_path, 'r', encoding='utf-8')]
            logging.info(f"self.cause_labels_context: {self.cause_labels_context}")
           
            # 载入通用法条、特殊法条内容: 按照标签位置依次排列。
            # load label2index map 
            dict_index2genLabel = {items.rstrip('\r\n').split('\t')[0]: items.rstrip('\r\n').split('\t')[1] for items in codecs.open(args.general_article_labelmap_path, 'r', encoding='utf-8')}
            dict_index2speLabel = {items.rstrip('\r\n').split('\t')[0]: items.rstrip('\r\n').split('\t')[1] for items in codecs.open(args.speific_article_labelmap_path, 'r', encoding='utf-8')}

            # load label2description map
            dict_label2genDescriptions = {items['no']: items['art'] for items in json.load(codecs.open(args.general_article_context_path, 'r', encoding='utf-8'))}
            dict_label2speDescriptions = {items['no']: items['art'] for items in json.load(codecs.open(args.speific_article_context_path, 'r', encoding='utf-8'))}

            self.gen_articles_context = [dict_label2genDescriptions[dict_index2genLabel[str(i)]].replace(" ", "")[:int(args.general_article_max_length / args.article_top_k)] for i in range(args.gen_article_num_classes)]
            self.spe_articles_context = [dict_label2speDescriptions[dict_index2speLabel[str(i)]].replace(" ", "")[:int(args.specific_article_max_length / args.article_top_k)] for i in range(args.spe_article_num_classes)]

            assert len(self.gen_articles_context) == args.gen_article_num_classes
 
             # 载入最终判决标签信息: 按照标签位置依次排列。
            self.fjp_labels_context = [i.rstrip('\r\n').split('\t')[1] for i in codecs.open(args.fjp_context_path, 'r', encoding='utf-8')]

            logging.info(f"单条法条最大字符长度为：{int(args.general_article_max_length / args.article_top_k)}")
            logging.info(f"self.gen_articles_context: {self.gen_articles_context[:3]}")
            logging.info
            (f"self.spe_articles_context: {self.spe_articles_context[:3]}")
            logging.info(f"self.fjp_labels_context: {self.fjp_labels_context}")
        else:
            raise NameError
        
        # 梯度更新参数层
        # self.trainable_param_names = ['layer.11', 'layer.10', 'layer.9', 'layer.8', 'layer.7', 'layer.6']
        # logging.info(f"self.trainable_param_names: {self.trainable_param_names}")
    
        # for name, param in self.model.named_parameters():
        #     if any(n in name for n in self.trainable_param_names):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

    def get_mask_token_embed(self, input_ids, sep_h):
        '''获取输入中[MASK]位置对应的表征'''
        # 获取所有 [MASK] 标记的位置
        mask_indices = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=False)
        #print(f"mask_indices: {mask_indices}")
    
        # 按照 batch_size 维度将索引分组
        batch_wise_indices = [[] for _ in range(sep_h.size(0))]
        for index in mask_indices:
            batch_wise_indices[index[0]].append(index[1])

        #print(f"batch_wise_indices: {batch_wise_indices}")
        batch_mask_embedding = []
        # 对每个 batch 中的索引列表执行相应操作，例如提取对应位置的 embedding
        for batch_index, indices in enumerate(batch_wise_indices):
            mask_embedding = sep_h[batch_index, indices, :]
            #print(f"1mask_embedding: {mask_embedding.shape}")
            batch_mask_embedding.append(mask_embedding.unsqueeze(0))
            #print(f"mask_embedding: {mask_embedding.shape}")
        
        batch_mask_embedding = torch.cat(batch_mask_embedding, dim=0)
        
        return batch_mask_embedding
            
    def forward(self, inputs_plaintiff=None,
                inputs_plea=None,
                inputs_defendant=None,
                inputs_fact=None,
                cause_label_id=None,
                gen_article_label_id=None,
                spe_article_label_id=None,
                fjp_labels_id=None):
        # 特殊符号表示：
        # [unused1]：原告诉称
        # [unused2]：原告诉请
        # [unused3]：被告辩称
        # [unused4]：事实描述
        # [unused5]：案由标签描述
        # [unused6]：通用法条描述
        # [unused7]：特定法条描述
        # [unused8]：最终判决标签描述
        if self.args.model_variants in ["LawformerCivilJP_baseCLS", "LawformerCivilJP_baseMean"]:
            '''Lawformer baseline: utilize CLS or global mean pool, which remove specific token in input.'''
            # LawformerCivilJP_baseCLS: baseline
            # LawformerCivilJP_baseMean: 未考虑任务依赖关系、结构感知、多视角特征的TML消融模型: TML(lawformer) wo TD、SADS
            # 子任务1：输入格式: [CLS] inputs_plaintiff inputs_plea inputs_defendant [SEP]
            input_ids_ccp, attention_masks_ccp = get_base_ccp_inputs(self.args, self.tokenizer, inputs_plaintiff, inputs_plea, inputs_defendant)

            # move data to device
            input_ids_ccp = input_ids_ccp.to(self.device)
            attention_masks_ccp = attention_masks_ccp.to(self.device)
            outputs_ccp = self.model(input_ids_ccp, attention_masks_ccp)

            # 子任务2：输入格式：[CLS] inputs_fact [SEP]
            input_ids_clap, attention_masks_clap = get_base_clap_inputs(self.args, self.tokenizer, inputs_fact)

            input_ids_clap = input_ids_clap.to(self.device)
            attention_masks_clap = attention_masks_clap.to(self.device)
            outputs_clap = self.model(input_ids_clap, attention_masks_clap)
            
            # 子任务3：输入格式：[CLS] inputs_plaintiff inputs_plea inputs_defendant inputs_fact [SEP]
            input_ids_fjp, attention_masks_fjp = get_base_fjp_inputs(self.args, self.tokenizer, inputs_plaintiff, inputs_plea, inputs_defendant, inputs_fact)
            input_ids_fjp = input_ids_fjp.to(self.device)
            attention_masks_fjp = attention_masks_fjp.to(self.device)
            outputs_fjp = self.model(input_ids_fjp, attention_masks_fjp)
        
            if self.args.model_variants == "LawformerCivilJP_baseCLS":
                pooler_output_ccp = outputs_ccp['pooler_output']
                pooler_output_clap = outputs_clap['pooler_output']
                pooler_output_fjp = outputs_fjp['pooler_output']
            elif self.args.model_variants == "LawformerCivilJP_baseMean":
                ccp_seq_h = outputs_ccp['last_hidden_state']
                clap_seq_h = outputs_clap['last_hidden_state']
                fjp_sep_h = outputs_fjp['last_hidden_state']

                pooler_output_ccp = remove_padding_avg(ccp_seq_h, attention_masks_ccp)
                pooler_output_clap = remove_padding_avg(clap_seq_h, attention_masks_clap)
                pooler_output_fjp = remove_padding_avg(fjp_sep_h, attention_masks_fjp)                
            else:
                raise NameError
            
            cause_logits = self.cause_cls_linear(pooler_output_ccp)
            gen_article_logits = self.gen_article_cls_linear(pooler_output_clap)
            spe_article_logits = self.spe_article_cls_linear(pooler_output_clap)
            fjp_logits = self.fjp_cls_linear(pooler_output_fjp)
            # train loss
            if cause_label_id is not None:
                cause_loss = self.CE_loss(cause_logits, cause_label_id)
        
                gen_article_loss = self.BCE_loss(torch.sigmoid(gen_article_logits), gen_article_label_id)
                spe_article_loss = self.BCE_loss(torch.sigmoid(spe_article_logits), spe_article_label_id)

                article_loss = self.args.gen_article_loss_weight * gen_article_loss + self.args.spe_article_loss_weight * spe_article_loss

                fjp_loss = self.CE_loss(fjp_logits, fjp_labels_id)          
    
                loss = self.args.cause_loss_weight * cause_loss + article_loss + self.args.fjp_loss_weight * fjp_loss
        
                return loss, cause_loss, article_loss, fjp_loss, cause_logits, gen_article_logits, spe_article_logits, fjp_logits

            return cause_logits, gen_article_logits, spe_article_logits, fjp_logits  

        # 未考虑任务依赖关系的TML消融模型: TML(lawformer) wo TD
        elif self.args.model_variants == "LawformerCivilJP":
            '''Multi-task learning of lawformer for civil judgment prediction.'''
            # 输入格式: [CLS][unused1] inputs_plaintiff [unused2] inputs_plea [unused3] inputs_defendant [SEP]
            input_ids_ccp, attention_masks_ccp = get_lawformer_ccp_inputs(self.args, self.tokenizer, inputs_plaintiff, inputs_plea, inputs_defendant)
    
            # 输入格式：[CLS] [unused4] inputs_fact [SEP]
            input_ids_clap, attention_masks_clap = get_lawformer_clap_inputs(self.args, self.tokenizer, inputs_fact)
    
            # 输入格式：[CLS] [unused1] inputs_plaintiff [unused2] inputs_plea [unused3] inputs_defendant [unused4] inputs_fact [SEP]
            input_ids_fjp, attention_masks_fjp = get_lawformer_fjp_inputs(self.args, self.tokenizer, inputs_plaintiff, inputs_plea, inputs_defendant, inputs_fact)

            # move data to device
            input_ids_ccp = input_ids_ccp.to(self.device)
            attention_masks_ccp = attention_masks_ccp.to(self.device)

            input_ids_clap = input_ids_clap.to(self.device)
            attention_masks_clap = attention_masks_clap.to(self.device)

            input_ids_fjp = input_ids_fjp.to(self.device)
            attention_masks_fjp = attention_masks_fjp.to(self.device)
            
            # 共用一个预训练语言模型进行文本编码
            outputs_ccp = self.model(input_ids_ccp, attention_masks_ccp)
            outputs_clap = self.model(input_ids_clap, attention_masks_clap)
            outputs_fjp = self.model(input_ids_fjp, attention_masks_fjp)

            seq_output_ccp = outputs_ccp['last_hidden_state']
            seq_output_clap = outputs_clap['last_hidden_state']
            seq_output_fjp = outputs_fjp['last_hidden_state']

            pooler_output_ccp = remove_padding_avg(seq_output_ccp, attention_masks_ccp)
            pooler_output_clap = remove_padding_avg(seq_output_clap, attention_masks_clap)
            pooler_output_fjp = remove_padding_avg(seq_output_fjp, attention_masks_fjp)

            cause_logits = self.cause_cls_linear(pooler_output_ccp)
            gen_article_logits = self.gen_article_cls_linear(pooler_output_clap)
            spe_article_logits = self.spe_article_cls_linear(pooler_output_clap)
            fjp_logits = self.fjp_cls_linear(pooler_output_fjp)
            
            # train loss
            if cause_label_id is not None:
                cause_loss = self.CE_loss(cause_logits, cause_label_id)
        
                gen_article_loss = self.BCE_loss(torch.sigmoid(gen_article_logits), gen_article_label_id)
                spe_article_loss = self.BCE_loss(torch.sigmoid(spe_article_logits), spe_article_label_id)

                article_loss = self.args.gen_article_loss_weight * gen_article_loss + self.args.spe_article_loss_weight * spe_article_loss

                fjp_loss = self.CE_loss(fjp_logits, fjp_labels_id)          
    
                loss = self.args.cause_loss_weight * cause_loss + article_loss + self.args.fjp_loss_weight * fjp_loss
        
                return loss, cause_loss, article_loss, fjp_loss, cause_logits, gen_article_logits, spe_article_logits, fjp_logits

            return cause_logits, gen_article_logits, spe_article_logits, fjp_logits  
        elif self.args.model_variants == "KnowAugLawformerCivilJP_TD":
            '''multi-task learning of lawformer for civil judgment prediction, considering task dependency.'''
            # 任务一，输入格式: [CLS][unused1] inputs_plaintiff [unused2] inputs_plea [unused3] inputs_defendant [SEP]
            input_ids_ccp, attention_masks_ccp = get_lawformer_ccp_inputs(self.args, self.tokenizer, inputs_plaintiff, inputs_plea, inputs_defendant)
            
            # cause prediction logits。CCP feature：global mean pooling + all_cause-aware feature
            # 输入id切片位置已check
            # move data to device
            input_ids_ccp = input_ids_ccp.to(self.device)
            attention_masks_ccp = attention_masks_ccp.to(self.device)
            outputs_ccp = self.model(input_ids_ccp, attention_masks_ccp)
            ccp_seq_h = outputs_ccp['last_hidden_state']

            # 参数意思：1表示根据列索引切片，第二个参数表示切片开始位置，第三个参数要切片的长度
            ccp_claim_h = ccp_seq_h.narrow(1, 0, self.args.plaintif_max_length)
            ccp_plea_h = ccp_seq_h.narrow(1, self.args.plaintif_max_length, self.args.plea_max_length)
            ccp_argument_h = ccp_seq_h.narrow(1, self.args.plaintif_max_length +  self.args.plea_max_length, self.args.defendant_max_length)

            ccp_claim_mask = attention_masks_ccp.narrow(1, 0, self.args.plaintif_max_length)
            ccp_plea_mask = attention_masks_ccp.narrow(1, self.args.plaintif_max_length, self.args.plea_max_length)
            ccp_argument_mask = attention_masks_ccp.narrow(1, self.args.plaintif_max_length +  self.args.plea_max_length, self.args.defendant_max_length)

            ccp_claim_mean_pool = remove_padding_avg(ccp_claim_h, ccp_claim_mask)
            ccp_plea_mean_pool = remove_padding_avg(ccp_plea_h, ccp_plea_mask)
            ccp_argument_mean_pool = remove_padding_avg(ccp_argument_h, ccp_argument_mask)

            ccp_classify_h = torch.cat([ccp_claim_mean_pool, ccp_plea_mean_pool, ccp_argument_mean_pool], dim=1) # B, 3h
            cause_logits = self.cause_cls_linear(ccp_classify_h)

            # 任务二clap：[CLS] [unused4] inputs_fact [unused5]top-K预测的案由标签信息 [SEP]
            input_ids_clap, attention_masks_clap = get_lawformerTCKE_clap_inputs(self.args, self.tokenizer, inputs_fact, self.cause_labels_context, cause_logits)

            # article prediction logits
            input_ids_clap = input_ids_clap.to(self.device)
            attention_masks_clap = attention_masks_clap.to(self.device)
            outputs_clap = self.model(input_ids_clap, attention_masks_clap)
            clap_seq_h = outputs_clap['last_hidden_state']

            # 参数意思：1表示根据列索引切片，第二个参数表示切片开始位置，第三个参数要切片的长度
            clap_fact_h = clap_seq_h.narrow(1, 0, self.args.fact_max_length)
            clap_topk_cause_h = clap_seq_h.narrow(1, self.args.fact_max_length, self.args.topk_cause_max_length) 

            clap_fack_mask = attention_masks_clap.narrow(1, 0, self.args.fact_max_length)
            clap_topk_cause_mask = attention_masks_clap.narrow(1, self.args.fact_max_length, self.args.topk_cause_max_length) 

            # 特征
            clap_fact_mean_pool = remove_padding_avg(clap_fact_h, clap_fack_mask)
            clap_topk_cause_mean_pool = remove_padding_avg(clap_topk_cause_h, clap_topk_cause_mask)

            clap_classify_h = torch.cat([clap_fact_mean_pool, clap_topk_cause_mean_pool], dim=1) # B, 2h

            gen_article_logits = self.gen_article_cls_linear(clap_classify_h)
            spe_article_logits = self.spe_article_cls_linear(clap_classify_h)

            # 任务三fjp：[CLS] [unused1] inputs_plaintiff [unused2] inputs_plea [unused3] inputs_defendant [unused4] inputs_fact [unused6] 通用Topk法条描述 [unused7] 特定Topk法条描述 [SEP]
            input_ids_fjp, attention_masks_fjp = get_lawformerTAKE_fjp_inputs(self.args, self.tokenizer, inputs_plaintiff, inputs_plea, inputs_defendant, inputs_fact, self.gen_articles_context, gen_article_logits, self.spe_articles_context, spe_article_logits)

            # fjp prediction logits
            input_ids_fjp = input_ids_fjp.to(self.device)
            attention_masks_fjp = attention_masks_fjp.to(self.device)
            outputs_fjp = self.model(input_ids_fjp, attention_masks_fjp)
            fjp_sep_h = outputs_fjp['last_hidden_state']

            # 参数意思：1表示根据列索引切片，第二个参数表示切片开始位置，第三个参数要切片的长度
            fjp_plaintiff_h = fjp_sep_h.narrow(1, 0, self.args.plaintif_max_length)
            fjp_plea_h = fjp_sep_h.narrow(1, self.args.plaintif_max_length, self.args.plea_max_length) 
            fjp_defendant_h = fjp_sep_h.narrow(1, self.args.plaintif_max_length+self.args.plea_max_length, self.args.defendant_max_length) 
            fjp_fact_h = fjp_sep_h.narrow(1, self.args.plaintif_max_length+self.args.plea_max_length+self.args.defendant_max_length, self.args.fact_max_length) 
            fjp_topk_gen_art_h = fjp_sep_h.narrow(1, self.args.plaintif_max_length+self.args.plea_max_length+self.args.defendant_max_length+self.args.fact_max_length, self.args.general_article_max_length) 
            fjp_topk_spe_art_h = fjp_sep_h.narrow(1, self.args.plaintif_max_length+self.args.plea_max_length+self.args.defendant_max_length+self.args.fact_max_length+self.args.general_article_max_length, self.args.specific_article_max_length) 

            # mask
            fjp_plaintiff_mask = attention_masks_fjp.narrow(1, 0, self.args.plaintif_max_length)
            fjp_plea_mask = attention_masks_fjp.narrow(1, self.args.plaintif_max_length, self.args.plea_max_length) 
            fjp_defendant_mask = attention_masks_fjp.narrow(1, self.args.plaintif_max_length+self.args.plea_max_length, self.args.defendant_max_length) 
            fjp_fact_mask = attention_masks_fjp.narrow(1, self.args.plaintif_max_length+self.args.plea_max_length+self.args.defendant_max_length, self.args.fact_max_length) 
            fjp_topk_gen_art_mask = attention_masks_fjp.narrow(1, self.args.plaintif_max_length+self.args.plea_max_length+self.args.defendant_max_length+self.args.fact_max_length, self.args.general_article_max_length) 
            fjp_topk_spe_art_mask = attention_masks_fjp.narrow(1, self.args.plaintif_max_length+self.args.plea_max_length+self.args.defendant_max_length+self.args.fact_max_length+self.args.general_article_max_length, self.args.specific_article_max_length) 

            fjp_claim_mean_pool = remove_padding_avg(fjp_plaintiff_h, fjp_plaintiff_mask) # B, h
            fjp_plea_mean_pool = remove_padding_avg(fjp_plea_h, fjp_plea_mask) # B, h
            fjp_argument_mean_pool = remove_padding_avg(fjp_defendant_h, fjp_defendant_mask) # B, h
            fjp_fact_mean_pool = remove_padding_avg(fjp_fact_h, fjp_fact_mask) # B, h
            fjp_topk_gen_art_mean_pool = remove_padding_avg(fjp_topk_gen_art_h, fjp_topk_gen_art_mask) # B, h
            fjp_topk_spe_art_mean_pool = remove_padding_avg(fjp_topk_spe_art_h, fjp_topk_spe_art_mask) # B, h
                        
            fjp_classify_h = torch.cat([fjp_claim_mean_pool, fjp_plea_mean_pool, fjp_argument_mean_pool, fjp_fact_mean_pool, fjp_topk_gen_art_mean_pool, fjp_topk_spe_art_mean_pool], dim=1)  # B, 6h

            fjp_logits = self.fjp_cls_linear(fjp_classify_h)

            if cause_label_id is not None:
                cause_loss = self.CE_loss(cause_logits, cause_label_id)

                gen_article_loss = self.BCE_loss(torch.sigmoid(gen_article_logits), gen_article_label_id)
                spe_article_loss = self.BCE_loss(torch.sigmoid(spe_article_logits), spe_article_label_id)
                article_loss = self.args.gen_article_loss_weight * gen_article_loss + self.args.spe_article_loss_weight * spe_article_loss

                fjp_loss = self.CE_loss(fjp_logits, fjp_labels_id)
            
                loss = self.args.cause_loss_weight * cause_loss + article_loss + self.args.fjp_loss_weight * fjp_loss

                return loss, cause_loss, article_loss, fjp_loss, cause_logits, gen_article_logits, spe_article_logits, fjp_logits

            return cause_logits, gen_article_logits, spe_article_logits, fjp_logits   
        else:
            raise NameError


