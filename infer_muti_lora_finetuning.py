# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer_lora_finetuning
import os
import torch
from deep_training.data_helper import ModelArguments, DataArguments
from transformers import HfArgumentParser, AutoConfig, GenerationConfig
from data_utils import train_info_args, NN_DataHelper, global_args, build_messages
from aigc_zoo.model_zoo.baichuan2.llm_model import MyTransformer,EffiArguments,\
    PromptArguments,BaichuanConfig,BaichuanTokenizer,LoraModel
from aigc_zoo.utils.llm_generate import Generate

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(train_info_args, allow_extra_keys=True)


    dataHelper = NN_DataHelper(model_args)
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(config_class_name=BaichuanConfig,
                                                              tokenizer_class_name=BaichuanTokenizer,)

    # 一般根据时间排序选最新的权重文件夹
    ckpt_dir = './best_ckpt/last'

    config = BaichuanConfig.from_pretrained(ckpt_dir)
    lora_args = EffiArguments.from_pretrained(ckpt_dir)

    assert lora_args.inference_mode == True

    new_num_tokens = config.vocab_size
    if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
        config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyTransformer(config=config, model_args=model_args, lora_args=lora_args,
                             torch_dtype=torch.float16,new_num_tokens=new_num_tokens,
                             # load_in_8bit=global_args["load_in_8bit"],
                             # # device_map="auto",
                             # device_map = {"":0} # 第一块卡
                             )

    # 加载lora权重
    pl_model.load_sft_weight(ckpt_dir)

    pl_model.eval().half().cuda()



    # backbone model replaced LoraModel
    lora_model: LoraModel = pl_model.backbone

    text_list = [
        "写一个诗歌，关于冬天",
        "晚上睡不着应该怎么办",
    ]

    # 基准模型推理
    with lora_model.disable_adapter():
        for input in text_list:
            # lora_model 调用子对象方法
            messages = build_messages(input)
            generation_config = GenerationConfig(max_new_tokens=512,
                                                 eos_token_id=config.eos_token_id,
                                                 pad_token_id=config.eos_token_id,
                                                 do_sample=True, top_k=5, top_p=0.85, temperature=0.3,
                                                 repetition_penalty=1.1, )
            response = lora_model.chat(tokenizer, messages=messages, generation_config=generation_config)
            print('input', input)
            print('output', response)

    lora_model.set_adapter(adapter_name='default')

    for input in text_list:
        # lora_model 调用子对象方法
        messages = build_messages(input)
        generation_config = GenerationConfig(max_new_tokens=512,
                                             eos_token_id=config.eos_token_id,
                                             pad_token_id=config.eos_token_id,
                                             do_sample=True, top_k=5, top_p=0.85, temperature=0.3,
                                             repetition_penalty=1.1, )
        response = lora_model.chat(tokenizer, messages=messages, generation_config=generation_config)
        print('input', input)
        print('output', response)