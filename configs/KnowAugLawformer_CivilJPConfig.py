import sys
import logging
sys.path.append('/mnt/sdb/leyuquan/github_backup/TML')
import configparser
import numpy as np
import torch as th


class Config(object):
    def __init__(self, file_path, dataset_name, local_rank, task):
        conf = configparser.ConfigParser()
        conf.read(file_path)
        
        # self.model_name = model_name
        self.dataset_name = dataset_name
        self.task = task
        self.local_rank = local_rank

        # General parameter
        self.learning_rate = conf.getfloat("General", "learning_rate")
        self.batch_size = conf.getint("General", "batch_size")
        self.accumulation_steps = conf.getint("General", "accumulation_steps")
        self.epochs = conf.getint("General", "epochs")
        self.input_max_length = conf.getint("General", "input_max_length")
        self.plaintif_max_length = conf.getint("General", "plaintif_max_length")
        self.defendant_max_length = conf.getint("General", "defendant_max_length")
        self.plea_max_length = conf.getint("General", "plea_max_length")
        self.fact_max_length = conf.getint("General", "fact_max_length")
        self.ccp_cause_max_length = conf.getint("General", "ccp_cause_max_length") 
        self.topk_cause_max_length = conf.getint("General", "topk_cause_max_length") # tok-k cause内容的最大长度
        self.general_article_max_length = conf.getint("General", "general_article_max_length")
        self.specific_article_max_length = conf.getint("General", "specific_article_max_length")
        self.all_general_article_name_max_length = conf.getint("General", "all_general_article_name_max_length")
        self.all_specific_article_name_max_length = conf.getint("General", "all_specific_article_name_max_length")
        self.all_fjp_name_max_length = conf.getint("General", "all_fjp_name_max_length") # fjp标签长度
        self.cause_question_prompt_length = conf.getint("General", "cause_question_prompt_length") # cause question prompt长度
        self.general_article_question_prompt_length = conf.getint("General", "general_article_question_prompt_length") # question prompt长度
        self.specific_article_question_prompt_length = conf.getint("General", "specific_article_question_prompt_length") # question prompt长度
        self.cause_top_k = conf.getint("General", "cause_top_k")
        self.article_top_k = conf.getint("General", "article_top_k")
        self.cause_num_classes = conf.getint("General", "cause_num_classes")
        self.gen_article_num_classes = conf.getint("General", "gen_article_num_classes")
        self.spe_article_num_classes = conf.getint("General", "spe_article_num_classes")
        self.fjp_num_classes = conf.getint("General", "fjp_num_classes")
        self.cause_loss_weight = conf.getint("General", "cause_loss_weight")
        self.gen_article_loss_weight = conf.getint("General", "gen_article_loss_weight")
        self.spe_article_loss_weight = conf.getint("General", "spe_article_loss_weight")
        self.fjp_loss_weight = conf.getint("General", "fjp_loss_weight")
        self.gpu_id = conf.getint("General", "gpu_id")
        self.random_seed = conf.getint("General", "random_seed")
        self.resume = conf.getboolean("General", "resume")
        self.is_train = conf.getboolean("General", "is_train")
        self.show_per_step = conf.getint("General", "show_per_step")
        self.model_name = conf.get("General", "model_name")
        self.model_variants = conf.get("General", "model_variants")
        self.cause_context_path = conf.get("General", "cause_context_path")
        self.fjp_context_path = conf.get("General", "fjp_context_path")
        self.general_article_context_path = conf.get("General", "general_article_context_path")
        self.speific_article_context_path = conf.get("General", "speific_article_context_path")
        self.general_article_labelmap_path = conf.get("General", "general_article_labelmap_path")
        self.speific_article_labelmap_path = conf.get("General", "speific_article_labelmap_path")
        self.is_metric2csv = conf.getboolean("General", "is_metric2csv")
        self.PTM_path = conf.get("General", "PTM_path")
        self.resume_checkpoint_path = conf.get("General", "resume_checkpoint_path")
        self.save_path = conf.get("General", "save_path")
        self.tensorboard_summary_writer_path = conf.get("General", "tensorboard_summary_writer_path")
        self.cause_metric_save_path = conf.get("General", "cause_metric_save_path")
        self.gen_articles_metric_save_path = conf.get("General", "gen_articles_metric_save_path")
        self.spe_articles_metric_save_path = conf.get("General", "spe_articles_metric_save_path")
        self.fjp_metric_save_path = conf.get("General", "fjp_metric_save_path")
        self.law_pred_save_path = conf.get("General", "law_pred_save_path")
        self.fjp_pred_save_path = conf.get("General", "fjp_pred_save_path")    

    def __repr__(self):
        '''show function'''
        logging.info(f'Parameters: {self.__dict__}')
        return '[Config Info]\tModel: {},\tTask: {},\tDataset: {}'.format(self.model_name, self.task, self.dataset_name)

