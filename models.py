import torch
from transformers import BartModel, BertModel, XLNetModel

class HuggingfaceModel(torch.nn.Module):
    def __init__(self, model_huggingface):
        super(HuggingfaceModel, self).__init__()
        self.model_huggingface = model_huggingface

    def forward(self, x):
        outputs_huggingface = self.model_huggingface(**x)
        return outputs_huggingface["logits"]

class BartModelForEnsemble(torch.nn.Module):
    def __init__(self, model_state_dict):
        super(BartModelForEnsemble, self).__init__()
        self.model = BartModel.from_pretrained("facebook/bart-base")
        self.model.load_state_dict(model_state_dict["model"])
    
    def forward(self, x):
        outputs = self.model(**x)
        mask_eos = x["input_ids"].eq(self.model.config.eos_token_id)
        hidden_state_eos = outputs["last_hidden_state"][:, mask_eos]
        return hidden_state_eos

class BertModelForEnsemble(torch.nn.Module):
    def __init__(self, model_state_dict):
        super(BertModelForEnsemble, self).__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.load_state_dict(model_state_dict["bert"])
    
    def forward(self, x):
        outputs = self.model(**x)
        hidden_state_cls = outputs["last_hidden_state"][0]
        return hidden_state_cls

class XLNetModelForEnsemble(torch.nn.Module):
    def __init__(self, model_state_dict):
        super(BertModelForEnsemble, self).__init__()
        self.model = XLNetModel.from_pretrained("xlnet-base-cased")
        self.model.load_state_dict(model_state_dict["transformer"])
    
    def forward(self, x):
        outputs = self.model(**x)
        hidden_state_cls = outputs["last_hidden_state"][-1]
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
            logits = model(**x[i])[0] # TODO: wrapper for models because of [0]
            list_logits.append(logits)
        tmp = torch.cat(list_logits, axis=1)
        logits = self.layer_linear(tmp)
        return logits
    
