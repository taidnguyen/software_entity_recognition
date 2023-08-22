"""
Self-regularization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForTokenClassification
from utils_nll import get_labels

def kl_div(p, q):
    return (p * ((p + 1e-5).log() - (q + 1e-5).log())).sum(-1)


class NERModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        label_list = get_labels(path=args.label_file)
        LABEL_TO_ID = {label: i for i, label in enumerate(label_list)}
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=args.num_class,
            hidden_dropout_prob=args.dropout_prob,
            label2id=LABEL_TO_ID,
            id2label={i: label for label, i in LABEL_TO_ID.items()}
        )
        self.model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, config=config)
        self.loss_fnt = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        if labels is not None:
            loss = self.loss_fnt(logits.view(-1, self.args.num_class), labels.view(-1))
            return loss, logits
        else:
            return logits


class NLLModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.models = nn.ModuleList()
        self.loss_fnt = nn.CrossEntropyLoss()
        self.device = [i % args.n_gpu for i in range(args.n_model)]
        model = NERModel(args).to(self.device[0])
        self.models.append(model)

    def forward(self, input_ids, attention_mask, labels=None):
        if labels is None:
            return self.models[0](input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  )
        else:
            num_models = self.args.n_model
            outputs = []
            for i in range(num_models):
                output = self.models[0](
                    input_ids=input_ids.to(self.device[0]),
                    attention_mask=attention_mask.to(self.device[0]),
                    labels=labels.to(self.device[0]) if labels is not None else None,
                )
                output = tuple([o.to(0) for o in output])
                outputs.append(output)
            model_output = outputs[0]
            loss = sum([output[0] for output in outputs]) / num_models
            logits = [output[1] for output in outputs]
            probs = [F.softmax(logit, dim=-1) for logit in logits]
            avg_prob = torch.stack(probs, dim=0).mean(0)
            mask = (labels.view(-1) != -1).to(logits[0])
            reg_loss = sum([kl_div(avg_prob, prob) * mask for prob in probs]) / num_models
            reg_loss = reg_loss.sum() / (mask.sum() + 1e-3)
            loss = loss + self.args.alpha_t * reg_loss
            model_output = (loss,) + model_output[1:]
        return model_output
