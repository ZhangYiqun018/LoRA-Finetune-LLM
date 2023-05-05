from train import Train
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, EvalPrediction, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM
from utils import get_config, get_tokenizer
import os
from evaluate import load
import numpy as np
from peft import PeftModel, prepare_model_for_int8_training

class Eval(Train):
    def __init__(self, config):
        super().__init__(config)
        self.lora_model_path = config['PATH']['lora_model_path']

    def load_model(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=self.model_path, device_map="auto", load_in_8bit=self.load_in_8bit)
        tokenizer = get_tokenizer(model_card=self.model_path)
        if self.load_in_8bit:
            model = prepare_model_for_int8_training(model)
        if self.do_lora:
            model = PeftModel.from_pretrained(
                model        = model,
                model_id     = self.lora_model_path,
                is_trainable = False,
            )

        return model, tokenizer
    def compute_rougeL(self, predictions, labels):
        rouge = load('rouge')
        results = rouge.compute(predictions=predictions, references=labels)
        
        return results
    
    def compute_bertscore(self, predictions, labels):
        bertscore = load("bertscore")

        result = bertscore.compute(predictions=predictions, references=labels, lang='en')
        # recall, f1, precision
        # calculate average
        recall    = np.average(result['recall'])
        f1        = np.average(result['f1'])
        precision = np.average(result['precision'])

        result['recall']    = recall
        result['f1']        = f1
        result['precision'] = precision
        return result
    
    def compute_bleu(self, predictions, labels):
        bleu = load("bleu")

        result = {}
        bleu_1 = bleu.compute(predictions=predictions, references=labels, max_order=1)
        result['bleu1'] = bleu_1['bleu']
        bleu_2 = bleu.compute(predictions=predictions, references=labels, max_order=2)
        result['bleu2'] = bleu_2['bleu']
        return result

    def compute_metrics(self, eval_pred: EvalPrediction):
        logits, labels = eval_pred

        # 替换-100
        logits = np.where(logits != -100, logits, self.tokenizer.pad_token_id)
        predictions = self.tokenizer.batch_decode(logits, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(logits, skip_special_tokens=True)

        result = dict()
        # metric1 bert_score
        result.update(self.compute_bertscore(predictions, labels))
        # metric2 bleu1,2
        result.update(self.compute_bleu(predictions, labels))
        # metric3 rouge-l
        result.update(self.compute_rougeL(predictions, labels))
        result['predictions'] = predictions
        result['labels'] = labels
        with open('result.txt', 'w') as w:
            w.write(str(result))
        return result   

    def init_evalargs(self):
        eval_args = Seq2SeqTrainingArguments(
            **self.train_args,
            predict_with_generate = True,
            prediction_loss_only  = False,
        )
        return eval_args

    def eval_loop(self):
        self.model, self.tokenizer = self.load_model()

        self.dataset = self.load_dataset()
        self.eval_args = self.init_evalargs()
        
        self.datacollator = DataCollatorForSeq2Seq(
            tokenizer          = self.tokenizer,
            return_tensors     = 'pt',
            padding            = "max_length",
            max_length         = 512,
            label_pad_token_id = self.tokenizer.pad_token_id,
        )

        valid_dataset = self.dataset['validation']

        trainer = Seq2SeqTrainer(
            model           = self.model,
            args            = self.eval_args,
            train_dataset   = None,
            eval_dataset    = valid_dataset,
            tokenizer       = self.tokenizer,
            data_collator   = self.datacollator,
            compute_metrics = lambda x: self.compute_metrics(x),
        )
        trainer.evaluate()


if __name__ == '__main__':
    config = get_config('config.json')
    os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA']['CUDA_VISIBLE_DEVICES']
    flan_t5_train = Eval(config=config)
    flan_t5_train.eval_loop()