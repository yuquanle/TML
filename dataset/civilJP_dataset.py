import sys
sys.path.append('/mnt/sdb/leyuquan/github_backup/TML')
from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader
from configs.Bert_CivilJPConfig import Config


__all__ = ['cpee']
class CivilJPDataset(Dataset):
    def __init__(self, mode='train', train_file=None, valid_file=None, test_file=None):
        assert mode in ['train', 'valid', 'test'], f"mode should be set to the one of ['train', 'valid', 'test']"
        self.mode = mode
        self.dataset = []
        if mode == 'train':
            self.dataset = self._load_data(train_file)
            print(f'Number of training dataset: {len(self.dataset)}')
        elif mode == 'valid':
            self.dataset = self._load_data(valid_file)
            print(f'Number of validation dataset: {len(self.dataset)}')
        else:
            self.dataset = self._load_data(test_file)
            print(f'Number of test dataset: {len(self.dataset)}.')

    def __getitem__(self, idx):
        idx = self.dataset[idx]['idx']
        plaintiff_text = self.dataset[idx]['plai'] 
        plea_text = self.dataset[idx]['plea'] # 可能一个或者多个诉求 
        defendant_text = self.dataset[idx]['defe'] 
        fact_text = self.dataset[idx]['fact'] 
        if self.mode in ['train', 'valid', 'test']:
            cause_label_id = self.dataset[idx]['cause_id'] 
            gen_article_label_id = self.dataset[idx]['gen_article_label_id'] 
            spe_article_label_id = self.dataset[idx]['spe_article_label_id'] 
            fjp_labels_id = self.dataset[idx]['fjp_labels_id']
            return idx, plaintiff_text, plea_text, defendant_text, fact_text, cause_label_id, gen_article_label_id, spe_article_label_id, fjp_labels_id
        else:
            raise NameError

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def collate_function(batch):
        idx, plaintiff_text, plea_text, defendant_text, fact_text, cause_label_id, gen_article_label_id, spe_article_label_id, fjp_labels_id = zip(*batch)
        return idx, plaintiff_text, plea_text, defendant_text, fact_text, cause_label_id, gen_article_label_id, spe_article_label_id, fjp_labels_id
    
    def _load_data(self, file_name):
        # 单案例可能包含数量不一致的诉求数量。
        dataset = []
        count = 0
        with open(file_name, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                count = count + 1
                if count == 100:
                    break
                json_dict = json.loads(line)
                plaintiff, pleas, defendant, fact, cause_label_id, gen_article_label_id, spe_article_label_id, fjp_labels_id = str(json_dict['plai']), json_dict['plea'], str(json_dict['defe']), json_dict['fact'], int(json_dict['cause_id']), json_dict['gen_article_id'], json_dict['spe_article_id'], json_dict['label']

                gen_article_label_id = [int(i) for i in gen_article_label_id]
                spe_article_label_id = [int(i) for i in spe_article_label_id]
                fjp_labels_id = [int(i) for i in fjp_labels_id]

                dataset.append({"idx": idx, "plai": plaintiff, "plea": pleas, "defe": defendant, "fact": fact, "cause_id": cause_label_id, "gen_article_label_id": gen_article_label_id, "spe_article_label_id": spe_article_label_id, "fjp_labels_id": fjp_labels_id})
        return dataset


if __name__ == '__main__':
    pass



