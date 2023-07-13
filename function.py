from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def create_model(continue_train, model_name_or_path, peft_model_id=None, 
                 get_pretrain_only=False, seq2seq=False, load_8bit_flag=False):

    if not get_pretrain_only:
        if seq2seq:
            peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1, bias='none')
        else:
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1, bias='none')
    else:
        peft_config = None
    
    if seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, load_in_8bit=load_8bit_flag)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_8bit=load_8bit_flag)

    if continue_train:
        print('Continue to train...')
        model = PeftModel.from_pretrained(model, peft_model_id)
        for name, param in model.named_parameters():
            if 'lora' in name or 'Lora' in name:
                param.requires_grad = True
    else:
        if not get_pretrain_only:
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        

    return model, peft_config
