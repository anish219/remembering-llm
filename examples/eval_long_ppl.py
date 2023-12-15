import torch
from tqdm import tqdm
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_metric
from torch.nn import CrossEntropyLoss
from streaming_llm.kv_cache import StartRecentKVCache
from streaming_llm.utils import parse_args, load

device = "cuda"

tokens = []
kvs = None

def main(args):
    global tokens
    global kvs

    data = load_dataset(args.dataset_name, split=args.split)

    model, tokenizer = load(args.model_name_or_path)

    nlls = []
    metrics = []
    loss_fn = CrossEntropyLoss(reduction="none")
    metric_fn = load_metric(args.dataset_name).compute
    past_key_values = None

    if args.enable_start_recent_kv_cache:
        if "llama" in model.config.model_type:
            k_seq_dim = v_seq_dim = 2
        elif "mpt" in model.config.model_type:
            v_seq_dim = 2
            k_seq_dim = 3
        elif "pythia" in model.config.model_type:
            k_seq_dim = v_seq_dim = 2
        elif "falcon" in model.config.model_type:
            v_seq_dim = 1
            k_seq_dim = 1
        else:
            raise ValueError(f"got {model.config.model_type}")
        kv_cache = StartRecentKVCache(
            start_size=args.start_size,
            recent_size=args.recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
        )
    else:
        kv_cache = None

    if args.enable_pos_shift:
        if "llama" in model.config.model_type:
            from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention

            enable_llama_pos_shift_attention(model)
        elif "falcon" in model.config.model_type:
            from streaming_llm.pos_shift.modify_falcon import (
                enable_falcon_pos_shift_attention,
            )

            enable_falcon_pos_shift_attention(model)
        elif "gpt_neox" in model.config.model_type:
            from streaming_llm.pos_shift.modify_gpt_neox import (
                enable_gpt_neox_pos_shift_attention,
            )

            enable_gpt_neox_pos_shift_attention(model)
        elif "mpt" in model.config.model_type:
            pass
        else:
            raise ValueError(f"got {model.config.model_type}")


    os.makedirs(args.output_dir, exist_ok=True)
    f = open(f"{args.output_dir}/log.txt", "w")

    num_eval_tokens = 0
    for item in data: # limit to num_samples somehow
        encodings = tokenizer(
            item['question'],
            item['context'],
            return_tensors='pt',
            max_length=512,
            truncation=True
        )

        print(encodings.input_ids[:, :10])

        seq_len = encodings.input_ids.size(1)
        print(f"seq_len: {seq_len}")
        pbar = tqdm(range(0, seq_len - 1))

        for idx in pbar:
            input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
            ids = input_ids.flatten().tolist()
            for i in range(len(ids)):
                tokens += (
                    tokenizer.decode(
                        ids[i:i+1],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                        spaces_between_special_tokens=False,
                    )
                    .strip()
                    .split(" ")
                )
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits.view(-1, model.config.vocab_size)
                past_key_values = outputs.past_key_values
                label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
                neg_log_likelihood = loss_fn(logits, label)
                metric_value = metric_fn(logits, label)
                if kv_cache is not None:
                    kvs = kv_cache.combine_kvs(kvs, kv_cache.get_new_kvs(past_key_values))
                    past_key_values = kv_cache(past_key_values)
            nlls.append(neg_log_likelihood)
            metrics.append(metric_value)
            pbar.set_description(
                f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}, metric: {metric_value.item():.2f}"
            )
            print(neg_log_likelihood.item(), file=f, flush=True)
            print(metric_value.item(), file=f, flush=True)
            num_eval_tokens += 1
            if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
                break
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break

    f.close()

    ppl = torch.exp(torch.stack(nlls).mean())
    print(ppl.item())
    with open(f"{args.output_dir}/ppl.txt", "w") as f:
        f.write(f"{ppl.item()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3"
    )
    parser.add_argument("--dataset_name", type=str, default="squad_v2")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--enable_start_recent_kv_cache", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2000)
    parser.add_argument("--enable_pos_shift", action="store_true")
    parser.add_argument("--output_dir", type=str, default="output/")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--num_eval_tokens", type=int, default=2000)
    args = parser.parse_args()

    main(args)