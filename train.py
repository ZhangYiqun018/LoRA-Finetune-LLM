from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, set_seed, EvalPrediction
from utils import get_config, get_dataset, get_model, get_tokenizer
import os
import datetime
from evaluate import load

class Train:
    def __init__(
        self,
        config,
    ):
        self.train_args = config['TRAIN']

        set_seed(config['TRAIN']['seed'])

        self.model_path = config['PATH']['model_path'] if config['PATH']['use_localmodel'] else config['PATH']['model_card']
        self.dataset_path = config['PATH']['dataset_path'] if config['PATH']['use_localdataset'] else config['PATH']['dataset_card']
        
        assert type(self.model_path) == str

        self.do_lora = config['LORA']['do_lora']
        self.load_in_8bit = config['LORA']['load_in_8bit']

        current_time = datetime.datetime.now().strftime("_%Y_%m_%d_%H")
        model_name = self.model_path.split('/')[-1]
        if self.load_in_8bit:
            model_name = '8bit_' + model_name
        if self.do_lora:
            model_name = 'lora_' + model_name

        self.train_args               = config['TRAIN']
        self.lora_args                = config['LORA']
        self.output_path              = os.path.join(config['PATH']['output_path'], model_name + current_time)
        self.train_args['output_dir'] = self.output_path
        self.train_args['run_name']   = model_name
        self.localdataset = config['PATH']['use_localdataset']
    
    def load_model(self):
        model      = get_model(model_card=self.model_path, do_lora=self.do_lora, lora_config=self.lora_args, load_in_8bit=self.load_in_8bit)
        tokenizer  = get_tokenizer(model_card=self.model_path)

        return model, tokenizer
    
    def load_dataset(self):
        dataset = get_dataset(data_path=self.dataset_path, tokenizer=self.tokenizer, local=self.localdataset)
        return dataset
    def init_trainargs(self):
        train_args = Seq2SeqTrainingArguments(
            **self.train_args,
            prediction_loss_only  = True,
            predict_with_generate = False,
        )
        return train_args
    
    def save_model(self):
        self.model.save_pretrained(self.output_path)  

    def preprocess_logits_for_metrics(self,):
        pass

    def train_loop(self):
        self.model, self.tokenizer = self.load_model()
        self.dataset    = self.load_dataset()

        self.train_args = self.init_trainargs()

        self.datacollator = DataCollatorForSeq2Seq(
            tokenizer          = self.tokenizer,
            return_tensors     = 'pt',
            padding            = "max_length",
            max_length         = 512,
            label_pad_token_id = self.tokenizer.pad_token_id,
        )

        train_dataset = self.dataset['train']
        valid_dataset = self.dataset['validation']

        trainer = Seq2SeqTrainer(
            model           = self.model,
            args            = self.train_args,
            train_dataset   = train_dataset,
            eval_dataset    = valid_dataset,
            tokenizer       = self.tokenizer,
            data_collator   = self.datacollator,
            compute_metrics = lambda x: self.compute_metrics(x),
        )
        
        trainer.train()

        self.save_model()

if __name__ == '__main__':
    config = get_config('config.json')

    os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA']['CUDA_VISIBLE_DEVICES']

    flan_t5_train = Train(config=config)

    flan_t5_train.train_loop()
        
