""" 
1. 模式与codet5的基本类似，但falcon可以自定义停止词，并根据停止词的
    的检测确定是否yield返回，而codet5用
    stopping_criteria=StoppingCriteriaList([CodeBlockStopper()])
    作为停止条件。
"""
import gc
from threading import Thread
from typing import Iterable

import torch
import transformers
from transformers import TextIteratorStreamer, GenerationConfig

from fastchat.utils import is_partial_stop

transformers.logging.set_verbosity_error()


@torch.inference_mode()
def generate_stream_falcon(
    model,
    tokenizer,
    params,
    device,
    context_len=2048,
    stream_interval=2,
    judge_sent_end=False,
):
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", 50))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    echo = bool(params.get("echo", True))
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    max_src_len = context_len - max_new_tokens - 8

    input_ids = input_ids[-max_src_len:]  # truncate from the left
    attention_mask = attention_mask[-max_src_len:]  # truncate from the left
    input_echo_len = len(input_ids)

    decode_config = dict(skip_special_tokens=True, clean_up_tokenization_spaces=True)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, **decode_config)

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=temperature >= 1e-5,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=10,
        top_p=top_p,
        top_k=top_k,
        eos_token_id=stop_token_ids,
    )

    generation_kwargs = dict(
        inputs=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        generation_config=generation_config,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    # echo表示记录输入的prompt,故rfind_start=len_prompt
    if echo:
        # means keep the prompt
        output = prompt
    else:
        output = ""

    for i, new_text in enumerate(streamer):
        output += new_text
        if i % stream_interval == 0:
            if echo:
                rfind_start = len_prompt
            else:
                rfind_start = 0

            partially_stopped = False
            # 如果规定了停止词，停止词可以是str也可以是str的列表
                # 在输出中从模型回答的位置用rfind检索停止词的位置，
                # 如果位置不为-1，只返回停止词之前的内容，但partially_stopped仍为False
                # 如果位置为-1，则调用is_partial_stop判断是否输出的尾部
                # len(stop_str)个字符是否包含stop_str的部分字符
                # 将判断的结果返回给partially_stopped,即如果检测到，partially_stopped为True
                # 如果没有检测到partially_stopped仍为False
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                    else:
                        # is_partial_stop：Check whether the output contains a partial stop str.
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # prevent yielding partial stop sequence
            # 如果output的尾部没有检测到停止词的部分字符，则yield结果
            #? 如果检测到呢？就不yield结果？
            if not partially_stopped:
                yield {
                    "text": output,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }
    output = output.strip()

    # finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif partially_stopped:
        finish_reason = None
    else:
        finish_reason = "stop"

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # clean
    gc.collect()
    torch.cuda.empty_cache()
