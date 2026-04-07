import torch
import argparse
import os
import json
from tqdm import tqdm
from transformers import set_seed
from qwen_vl_utils import process_vision_info
import pandas as pd


# 很多都是低质量数据而非hard

def build_inference_batch(questions, image_folder, batch_size=1):
    n_batches = len(questions) // batch_size + (1 if len(questions) % batch_size > 0 else 0)

    batches = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(questions))
        batch = questions[start_idx:end_idx]
        batch = deal_one_batch(batch, image_folder)
        batches.append(batch)
    return batches


def deal_one_batch(batch, image_folder):
    new_batch = []
    item_batch = []
    for item in batch:
        image_name = item["image"]
        question = item["conversations"][0]['value']
        image_path = os.path.join(image_folder, image_name)

        # ground_truth = item["answer_one"]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        new_batch.append(messages)
        item_batch.append(item)
    return (new_batch, item_batch)




def main(args):
    if args.adapter_mode == 'self+cross_share_head':
        from train.qwen_vl_re_see_vision_old_param_null_token_share_head.modeling_qwen_vl import Qwen2_5_VLForConditionalGeneration
        from train.qwen_vl_re_see_vision_old_param_null_token_share_head.Qwen2_5_VLProcessor import Qwen2_5_VLProcessor
    else:
        assert False

    checkpoint = args.checkpoint

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attention_mode,
            device_map="cuda",
        )
    processor = Qwen2_5_VLProcessor.from_pretrained(checkpoint)

    processor.tokenizer.padding_side = 'left'

    model.config.my_decoder_mode = args.my_decoder_mode



    image_folder = '/data/dongxinpeng/datasets/'
    question_file = f'/data/dongxinpeng/datasets/ShareGPT4V/my_sharegpt4v_instruct_check_grpo_short.json'
    # questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
    with open(question_file, 'r', encoding='utf-8') as f:
        questions = json.load(f) # 使用 json.load() 直接从文件对象加载

    # print(f"Total questions: {len(questions)}")
    # print(questions[0])

    # assert False

    # output_folder = '/data/dongxinpeng/datasets/my_train_data'
    # os.makedirs(output_folder, exist_ok=True)

    total_questions = len(questions)
    
    
    wrong_data_file_path = '/data/dongxinpeng/my_former_work/work/instruct_wrong.jsonl'
    
    wrong_data_file = open(wrong_data_file_path, "w")
    
    
    batchs = build_inference_batch(questions, image_folder, batch_size=args.batch_size)


    for batch in tqdm(batchs):
        massages, item_batch = batch

        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in massages
        ]
        # print(text)
    
        image_inputs, video_inputs = process_vision_info(massages)
        # print(image_inputs)  # [<PIL.Image.Image image mode=RGB size=644x420 at 0x7F28FC579600>]  

        # assert False

        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(model.device)

        

        # generated_ids = model.generate(**inputs, max_new_tokens=1024)

        generated_ids = model.generate(**inputs, 
                                   max_new_tokens=128,
                                   temperature=args.temperature,  # Increased from 0.7
                                   top_k=args.top_k,         # Added top_k sampling
                                   top_p=args.top_p,       # Added nucleus sampling
                                   do_sample=True ,   # Enable sampling
                                   repetition_penalty=1.05,)


        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        out_ans = output_text[0]
        out_ans = out_ans.lower().replace('.', '')
        
        
        for item, out_ans in zip(item_batch, output_text):
            
            ground_truth = item["conversations"][1]['value'].lower()
            out_ans = out_ans.lower().replace('.', '')
            
            # print(item)
            # assert False
            if ground_truth in out_ans:
                pass
            else:
                item['out_put'] = out_ans
            
                wrong_data_file.write(json.dumps(item) + "\n")

                wrong_data_file.flush()

        # assert False

    return








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)

    parser.add_argument("--save_id", type=str, default=None)

    parser.add_argument("--attention_mode", type=str, default='eager')
  
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.001)
    parser.add_argument("--top_k", type=int, default=1)

    parser.add_argument("--no_output", action='store_true', default=False)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--my_decoder_mode", type=str, default=None)
    parser.add_argument("--train_param", type=str, default=None)
    
    parser.add_argument("--adapter_mode", type=str, default='self+cross_share_head')
    
    parser.add_argument("--batch_size", type=int, default=1)
    
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)