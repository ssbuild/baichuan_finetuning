# -*- coding: utf-8 -*-
# @Time:  23:20
# @Author: tk
# @File：model_maps
from aigc_zoo.constants.define import (TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING)

__all__ = [
    "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING",
    "train_model_config"
]

train_info_models = {
    # 第二代模型
    'Baichuan2-7B-Base': {
        'model_type': 'baichuan',
        'model_name_or_path': '/data/nlp/pre_models/torch/baichuan2/Baichuan2-7B-Base',
        'config_name': '/data/nlp/pre_models/torch/baichuan2/Baichuan2-7B-Base/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/baichuan2/Baichuan2-7B-Base',
    },
    'Baichuan2-7B-Chat': {
        'model_type': 'baichuan',
        'model_name_or_path': '/data/nlp/pre_models/torch/baichuan2/Baichuan2-7B-Chat',
        'config_name': '/data/nlp/pre_models/torch/baichuan2/Baichuan2-7B-Chat/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/baichuan2/Baichuan2-7B-Chat',
    },

    'Baichuan2-13B-Base': {
        'model_type': 'baichuan',
        'model_name_or_path': '/data/nlp/pre_models/torch/baichuan2/Baichuan2-13B-Base',
        'config_name': '/data/nlp/pre_models/torch/baichuan2/Baichuan2-13B-Base/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/baichuan2/Baichuan2-13B-Base',
    },
    'Baichuan2-13B-Chat': {
        'model_type': 'baichuan',
        'model_name_or_path': '/data/nlp/pre_models/torch/baichuan2/Baichuan2-13B-Chat',
        'config_name': '/data/nlp/pre_models/torch/baichuan2/Baichuan2-13B-Chat/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/baichuan2/Baichuan2-13B-Chat',
    },

    'baichuan2-7b-chat-int4': {
        'model_type': 'baichuan',
        'model_name_or_path': '/data/nlp/pre_models/torch/baichuan2/baichuan2-7b-chat-int4',
        'config_name': '/data/nlp/pre_models/torch/baichuan2/baichuan2-7b-chat-int4/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/baichuan2/baichuan2-7b-chat-int4',
    },

    'baichuan2-13b-chat-int4': {
        'model_type': 'baichuan',
        'model_name_or_path': '/data/nlp/pre_models/torch/baichuan2/baichuan2-13b-chat-int4',
        'config_name': '/data/nlp/pre_models/torch/baichuan2/baichuan2-13b-chat-int4/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/baichuan2/baichuan2-13b-chat-int4',
    },

    # 第一代

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


# 按需修改
# TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING

train_model_config = train_info_models['Baichuan2-7B-Chat']