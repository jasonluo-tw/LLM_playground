from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def create_model(continue_train, model_name_or_path, peft_model_id=None):
    peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1, bias='none')
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    #model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    if continue_train:
      print('Continue to train...')
      model = PeftModel.from_pretrained(model, peft_model_id)
      for name, param in model.named_parameters():
        if 'lora' in name or 'Lora' in name:
            param.requires_grad = True
    else:
      model = get_peft_model(model, peft_config)
    
    model.print_trainable_parameters()

    return model, peft_config