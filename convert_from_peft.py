# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/21 16:51


import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser, GenerationConfig
from data_utils import train_info_args, NN_DataHelper, global_args, build_messages
from aigc_zoo.model_zoo.baichuan2.llm_model import MyTransformer,LoraModel,LoraArguments,BaichuanConfig,BaichuanTokenizer

# 加载 peft 权重
if __name__ == '__main__':
    ckpt_dir = './peft_lora'  # peft lora 路径
    output_lora_dir = './lora'
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(train_info_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args, )
    tokenizer: BaichuanTokenizer
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=BaichuanTokenizer, config_class_name=BaichuanConfig)

    # new_num_tokens = config.vocab_size
    # if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
    #     config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyTransformer(config=config, model_args=model_args,
                             torch_dtype=torch.float16,
                             # new_num_tokens=new_num_tokens,#扩充词
                             # load_in_8bit=global_args["load_in_8bit"],
                             # # device_map="auto",
                             # device_map = {"":0} # 第一块卡
                             )
    # 加载peft权重
    pl_model.load_peft_weight(ckpt_dir,is_trainable=False)
    #保存权重
    pl_model.save_sft_weight(output_lora_dir)

    model = pl_model.get_llm_model()

    model.eval().half().cuda()
    text_list = ["写一个诗歌，关于冬天",
                 "晚上睡不着应该怎么办",
                 "从南京到上海的路线",
                 ]

    for input in text_list:
        messages = build_messages(input)
        generation_config = GenerationConfig(eos_token_id=config.eos_token_id,
                                             pad_token_id=config.eos_token_id,
                                             do_sample=True, top_k=5, top_p=0.85, temperature=0.3,
                                             repetition_penalty=1.1, )
        response = model.chat(tokenizer, messages=messages, generation_config=generation_config)
        print('input', input)
        print('output', response)