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

        model_path = config['PATH']['model_path']
        dataset_path = config['PATH']['dataset_path'] if config['PATH']['local'] else config['PATH']['dataset_card']
        do_lora = config['LORA']['do_lora']

        current_time = datetime.datetime.now().strftime("_%Y_%m_%d_%H")
        model_name = model_path.split('/')[-1]
        if do_lora:
            model_name = 'lora_' + model_name

        self.train_args               = config['TRAIN']
        self.lora_args                = config['LORA']
        self.output_path              = os.path.join(config['PATH']['output_path'], model_name + current_time)
        self.train_args['output_dir'] = self.output_path
        
        self.model      = get_model(model_card=model_path, do_lora=do_lora, lora_config=self.lora_args)
        self.tokenizer  = get_tokenizer(model_card=model_path)
        self.dataset    = get_dataset(data_path=dataset_path, tokenizer=self.tokenizer, local=config['PATH']['local'])
        
    def init_trainargs(self):
        train_args = Seq2SeqTrainingArguments(
            **self.train_args,
        )
        return train_args
    
    def save_model(self):
        self.model.save_pretrained(self.output_path)

    def compute_metrics(self, eval_pred: EvalPrediction):
        logits, labels = eval_pred
        print(logits.shape)
        predictions = self.tokenizer.batch_decode(logits)
        labels = self.tokenizer.batch_decode(logits)

        bertscore = load("bertscore")

        result = bertscore.compute(predictions=predictions, references=labels, lang='en')
        return result      

    def preprocess_logits_for_metrics(self,):
        pass

    def loop(self):
        self.train_args = self.init_trainargs()

        self.datacollator = DataCollatorForSeq2Seq(
            tokenizer      = self.tokenizer,
            return_tensors = 'pt',
            padding        = True,
        )

        train_dataset = self.dataset['train'].select(range(100))
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

    flan_t5_train.init_trainargs()

    flan_t5_train.loop()
        
