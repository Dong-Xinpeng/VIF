import os
import torch
from peft import LoraConfig
import ast
import pathlib
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, HfArgumentParser

from train.trainer_grpo import QwenGRPOTrainer
# from train.data import make_supervised_data_module
from train.data_grpo import make_grpo_data_module
from train.params import DataArguments, ModelArguments, GRPOArguments
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
from monkey_patch_forward import replace_qwen2_5_with_mixed_modality_forward, replace_qwen_2_with_mixed_modality_forward
from monkey_patch_vision import replace_qwen2_5_vision
from src.utils import  load_reward_funcs
import deepspeed



local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)
    
    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

def configure_llm(model, training_args):
    # lm_head = model.lm_head.parameters()
    # set_requires_grad(lm_head, not training_args.freeze_llm)
    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def configure_llm_head(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm_head)
    # print('lm_head:',model.lm_head)


def configure_inf_former(model, training_args):
    
    if 'share_head' not in training_args.adapter_mode:
        lm_head_former = model.lm_head_former.parameters()
        set_requires_grad(lm_head_former, not training_args.freeze_inf_former)  # TODO:


    inference_former_params = model.inference_former.parameters()
    set_requires_grad(inference_former_params, not training_args.freeze_inf_former)

    norm_former_params = model.norm_former.parameters()
    set_requires_grad(norm_former_params, not training_args.freeze_inf_former)



def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, GRPOArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    
    
    train_param = 'train_param:'
    if not training_args.freeze_llm:
        train_param = train_param + '+llm'
    if not training_args.freeze_llm_head:
        train_param = train_param + '+llm_head'
    if not training_args.freeze_vision_tower:
        train_param = train_param + '+vision_tower'
    if not training_args.freeze_merger:
        train_param = train_param + '+merger'
    if (not training_args.freeze_inf_former) and (training_args.adapter_mode != 'no_adapter'):
        train_param = train_param + '+inf_former'
    
    training_args.output_dir = os.path.join(training_args.output_dir,train_param)
    
    if training_args.adapter_mode == 'self+cross':
        from qwen_vl_re_see_vision.modeling_qwen_vl import Qwen2_5_VLForConditionalGeneration
    elif training_args.adapter_mode == 'two_cross':
        from qwen_vl_re_see_vision_two_cross.modeling_qwen_vl import Qwen2_5_VLForConditionalGeneration
    elif training_args.adapter_mode == 'no_adapter':
        from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
    elif training_args.adapter_mode == 'self+cross_old':
        from train.qwen_vl_re_see_vision_old_param_null_token.modeling_qwen_vl import Qwen2_5_VLForConditionalGeneration
    elif training_args.adapter_mode == 'self+cross_share_head':
        from train.qwen_vl_re_see_vision_old_param_null_token_share_head.modeling_qwen_vl import Qwen2_5_VLForConditionalGeneration
    else:
        assert False
        
        
    # if deepspeed.comm.get_rank() == 0:    
    #     print("training_args.adapter_mode:",training_args.adapter_mode)
    
    
    
    
    
    training_args.use_liger_loss = False
    # if "Qwen2.5" in model_args.model_id:
    #     # monkey patch the vision model
    #     replace_qwen2_5_vision()
    #     # It monkey patches the forward to handle mixed modality inputs.
    #     replace_qwen2_5_with_mixed_modality_forward(use_liger=False)
    # else:
    #     # It monkey patches the forward to handle mixed modality inputs.
    #     replace_qwen_2_with_mixed_modality_forward(use_liger=False)

    if data_args.nframes is not None and data_args.fps is not None:
        raise ValueError("You cannot set both `nframes` and `fps` at the same time. Please set only one of them.")

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    if "Qwen2.5" in model_args.model_id:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
            **bnb_model_from_pretrained_args
        )

    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
            **bnb_model_from_pretrained_args
        )

    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_llm_head(model_to_configure, training_args)
    if training_args.adapter_mode != 'no_adapter':
        configure_inf_former(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    if training_args.bits in [4,8]:
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": True})
    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    peft_config = None



    processor = AutoProcessor.from_pretrained(model_args.model_id)

   

    dataset_module = make_grpo_data_module(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args)
    
    
    if 'share_head' not in training_args.adapter_mode:
    # if training_args.adapter_mode != 'no_adapter':
        # 收集 self.lm_head 和 self.lm_head_former 的权重
        with deepspeed.zero.GatheredParameters([model.lm_head.weight, model.lm_head_former.weight], modifier_rank=0):
            # 只有 rank 0 负责执行复制操作
            if deepspeed.comm.get_rank() == 0: # 确保只有主进程（rank 0）执行复制操作
                print("Rank 0: Performing weight copy...")
                # 复制权重数据
                model.lm_head_former.weight.data.copy_(model.lm_head.weight.data)
                print("Rank 0: Weight copy complete.")
            deepspeed.comm.barrier() # 确保所有进程等待 rank 0 完成复制和分发
    
    reward_funcs = load_reward_funcs("src.train.reward_funcs")
    # check
    
    # print(training_args)
    # print(model_args)
    # print(data_args)
    # if deepspeed.comm.get_rank() == 0:
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             print(f'{name} trainable')
    #         else:
    #             print(f'{name} frozen')
    # assert False

    

    trainer = QwenGRPOTrainer(
        adapter_mode=training_args.adapter_mode,
        model=model,
        train_dataset=dataset_module["train_dataset"],
        eval_dataset = dataset_module["eval_dataset"],
        reward_funcs=reward_funcs,
        args=training_args,
        peft_config=peft_config,
    )
    
    # if deepspeed.comm.get_rank() == 0:
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             print(f'{name} trainable')
    #         else:
    #             print(f'{name} frozen')
    # assert False

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()