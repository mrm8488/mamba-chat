import torch
import argparse

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, TrainingArguments
from trainer.data import ChatDataModule
from trainer.mamba_trainer import MambaTrainer

chat_template_id = "HuggingFaceH4/zephyr-7b-beta"


def run(args):
        
    model = MambaLMHeadModel.from_pretrained(args.model, dtype=torch.bfloat16, device="cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = AutoTokenizer.from_pretrained(chat_template_id).chat_template


    train_data_module = ChatDataModule(
        tokenizer=tokenizer,
        data_path=args.train_data_path,
        conversation_template=tokenizer.chat_template,
        max_tokens=args.max_tokens
    )
    if args.do_eval:
        test_data_module = ChatDataModule(
        tokenizer=tokenizer,
        data_path=args.test_data_path,
        conversation_template=tokenizer.chat_template,
        max_tokens=args.max_tokens
    )
                

    trainer = MambaTrainer(
        model=model,
        train_dataset=train_data_module.dataset,
        eval_dataset=test_data_module.dataset if args.do_eval else None,
        tokenizer=tokenizer,
        args=TrainingArguments(
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim=args.optim,
            output_dir=args.output_dir,
            logging_steps=50,
            save_steps=500,    
            save_total_limit=args.save_total_limit,
            do_eval=args.do_eval,
            eval_steps=args.eval_steps
        ),
        data_collator=train_data_module.data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="state-spaces/mamba-2.8b")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--max_tokens", type=int, default=2048)    
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--train_data_path", type=str, default="./data/ultrachat_small.jsonl")
    parser.add_argument("--test_data_path", type=str, required=False)    
    parser.add_argument("--num_epochs", type=float, default=1)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=500)    
    parser.add_argument("--output_dir", type=str, default="mamba-finetuned")
    parser.add_argument("--save_total_limit", type=int, default=2)    
    parser.add_argument("--resume_from_checkpoint", action="store_true")     
    args = parser.parse_args()

    run(args)
