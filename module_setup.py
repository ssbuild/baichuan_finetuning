# -*- coding: utf-8 -*-
# @Time:  23:34
# @Author: tk
# @File：model_setup
from deep_training.utils.hf import register_transformer_model,register_transformer_config
from config import *

_model_card = train_info_args["model_name_or_path"].split('/')[-1].lower()
assert "7b" in _model_card or "13b" in _model_card
if "baichuan2" in _model_card:

    if "7b" in _model_card:
        from aigc_zoo.model_zoo.baichuan.baichuan2_7b.llm_model import (MyTransformer,PetlArguments, # noqa
                                                                        LoraConfig,PetlModel,
                                                                        PromptArguments,
                                                                        BaichuanConfig,
                                                                        BaichuanTokenizer)
    else:
        from aigc_zoo.model_zoo.baichuan.baichuan2_13b.llm_model import (MyTransformer, PetlArguments, # noqa
                                                                        LoraConfig,PetlModel,
                                                                        PromptArguments,
                                                                        BaichuanConfig,
                                                                        BaichuanTokenizer)
else:
    if "7b" in _model_card:
        from aigc_zoo.model_zoo.baichuan.baichuan_7b.llm_model import (MyTransformer, PetlArguments,  # noqa
                                                                        LoraConfig, PetlModel,
                                                                        PromptArguments,
                                                                        BaichuanConfig,
                                                                        BaichuanTokenizer)
    else:
        from aigc_zoo.model_zoo.baichuan.baichuan_13b.llm_model import (MyTransformer, PetlArguments,  # noqa
                                                                         LoraConfig, PetlModel,
                                                                         PromptArguments,
                                                                         BaichuanConfig,
                                                                         BaichuanTokenizer)

del _model_card

__all__ = [
    "module_setup"
]

def module_setup():
    pass
    # 导入模型
    #register_transformer_config(XverseConfig)
    # register_transformer_model(LlamaForCausalLM, AutoModelForCausalLM)