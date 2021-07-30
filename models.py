# this file contains various models used in the training and inference scripts

import torch
from transformers import AutoModelForSequenceClassification

class HuggingfaceModel(torch.nn.Module):
    def __init__(self, model_huggingface):
        super(HuggingfaceModel, self).__init__()
        self.model_huggingface = model_huggingface

    def forward(self, x):
        outputs_huggingface = self.model_huggingface(**x)
        return outputs_huggingface["logits"]

class BartModelForEnsemble(torch.nn.Module):
    def __init__(self, model_state_dict, tokenizer=None):
        super(BartModelForEnsemble, self).__init__()
        model_huggingface = AutoModelForSequenceClassification.from_pretrained("facebook/bart-base", num_labels=2)
        if tokenizer is not None:
            model_huggingface.resize_token_embeddings(len(tokenizer))
        self.model = HuggingfaceModel(model_huggingface)
        self.model.load_state_dict(model_state_dict)
        # self.model = HuggingfaceModel(model_huggingface).model_huggingface.model
    
    def forward(self, input_ids):
        outputs = self.model(**x)
        mask_eos = input_ids["input_ids"].eq(self.model.config.eos_token_id)
        hidden_state_eos = outputs["last_hidden_state"][mask_eos]
        return hidden_state_eos

class BertModelForEnsemble(torch.nn.Module):
    def __init__(self, model_state_dict, tokenizer=None):
        super(BertModelForEnsemble, self).__init__()
        model_huggingface = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        if tokenizer is not None:
            model_huggingface.resize_token_embeddings(len(tokenizer))
        self.model = HuggingfaceModel(model_huggingface)
        self.model.load_state_dict(model_state_dict)
        
    
    def forward(self, x):
        outputs = self.model(**x)
        hidden_state_cls = outputs["last_hidden_state"][:,0]
        return hidden_state_cls

class BertweetModelForEnsemble(torch.nn.Module):
    def __init__(self, model_state_dict, tokenizer=None):
        super(BertweetModelForEnsemble, self).__init__()
        model_huggingface = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=2)
        if tokenizer is not None:
            model_huggingface.resize_token_embeddings(len(tokenizer))
        self.model = HuggingfaceModel(model_huggingface)
        self.model.load_state_dict(model_state_dict)

    def forward(self, x):
        outputs = self.model(**x)
        hidden_state_cls = outputs["last_hidden_state"][:,0]
        return hidden_state_cls

class XLNetModelForEnsemble(torch.nn.Module):
    def __init__(self, model_state_dict, tokenizer=None):
        super(XLNetModelForEnsemble, self).__init__()
        model_huggingface = AutoModelForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=2)
        if tokenizer is not None:
            model_huggingface.resize_token_embeddings(len(tokenizer))
        self.model = HuggingfaceModel(model_huggingface)
        self.model.load_state_dict(model_state_dict)
    
    def forward(self, x):
        outputs = self.model(**x)
        hidden_state_cls = outputs["last_hidden_state"][:,-1]
        return hidden_state_cls

class EnsembleModel(torch.nn.Module):
    def __init__(self, list_models, freeze_models=False, size_hidden_state=2):
        super(EnsembleModel, self).__init__()
        self.list_models = torch.nn.ModuleList(list_models)
        self.layer_linear = torch.nn.Linear(
             in_features=len(list_models) * size_hidden_state,
             out_features=2,
        )
        if freeze_models:
            for model in self.list_models:
                for param in model.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        list_logits = []
        for i in range(len(self.list_models)):
            model = self.list_models[i]
            logits = model(x[i])
            list_logits.append(logits)
        tmp = torch.cat(list_logits, axis=1)
        logits = self.layer_linear(tmp)
        return logits
    
