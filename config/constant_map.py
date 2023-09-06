# -*- coding: utf-8 -*-
# @Time:  23:20
# @Author: tk
# @File：model_maps

train_info_models = {
    'Baichuan-13B-Base': {
        'model_type': 'baichuan',
        'model_name_or_path': '/data/nlp/pre_models/torch/baichuan/Baichuan-13B-Base',
        'config_name': '/data/nlp/pre_models/torch/baichuan/Baichuan-13B-Base/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/baichuan/Baichuan-13B-Base',
    },
    'Baichuan-13B-Chat': {
        'model_type': 'baichuan',
        'model_name_or_path': '/data/nlp/pre_models/torch/baichuan/Baichuan-13B-Chat',
        'config_name': '/data/nlp/pre_models/torch/baichuan/Baichuan-13B-Chat/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/baichuan/Baichuan-13B-Chat',
    },
    'baichuan-13b-chat-int4': {
        'model_type': 'baichuan',
        'model_name_or_path': '/data/nlp/pre_models/torch/baichuan/baichuan-13b-chat-int4',
        'config_name': '/data/nlp/pre_models/torch/baichuan/baichuan-13b-chat-int4/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/baichuan/baichuan-13b-chat-int4',
    },


}


# 'target_modules': ['query_key_value'],  # bloom,gpt_neox
# 'target_modules': ["q_proj", "v_proj"], #llama,opt,gptj,gpt_neo
# 'target_modules': ['c_attn'], #gpt2
# 'target_modules': ['project_q','project_v'] # cpmant

train_target_modules_maps = {
    'baichuan': ['W_pack'],
    'moss': ['qkv_proj'],
    'chatglm': ['query_key_value'],
    'bloom' : ['query_key_value'],
    'gpt_neox' : ['query_key_value'],
    'llama' : ["q_proj", "v_proj"],
    'opt' : ["q_proj", "v_proj"],
    'gptj' : ["q_proj", "v_proj"],
    'gpt_neo' : ["q_proj", "v_proj"],
    'gpt2' : ['c_attn'],
    'cpmant' : ['project_q','project_v'],
    'rwkv' : ['key','value','receptance'],
}


train_model_config = train_info_models['Baichuan-13B-Chat']