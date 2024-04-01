import json
import os
import argparse
from typing import Dict, Optional, Union
from autogen.agentchat import AssistantAgent
from datasets import load_dataset
from mnms.constants import *
from mnms.agent import MnmsUserAgent, MnmsUserAgentLocal
from mnms.prompt import FeedbackPrompt, JsonGenPrompt, ReACTPrompt, CodeGenPrompt
from mnms.parser import Parser, CodeParser
from mnms.verifier import Verifier, CodeVerifier
from mnms.execution.executor import Executor, CodeExecutor
from mnms.execution.config import *



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-mode', default='multi-step', type=str, help="use global or local planning mode", choices=['multi-step', 'step-by-step'])
    parser.add_argument('--action-format', default='json', type=str, help="use dict or code as the format of the action", choices=['json', 'code'])
    parser.add_argument('--model', default='gpt-4-1106-preview', type=str, help="a string representing a unique model.")
    parser.add_argument('--verify', action='store_true', help="whether to use feedback from verification or not.")
    parser.add_argument('--execute', action='store_true', help="whether to use feedback from execution or not.")
    parser.add_argument('--max-reply', default=0, type=int, help="the maximum number of replies after the first attempt.")
    parser.add_argument('--seed', default=42, type=int, help="the random seed used in llm prediction.")
    parser.add_argument('--input-file', default=None, type=str, help="the file to import the input from.")
    parser.add_argument('--exp-id', default=None, type=str, help="a unique string for identifying the current experiment.")
    parser.add_argument('--simulate', action='store_true', help="only print the initial planning prompt instead of actually running planning agent.")
    parser.add_argument('--output-dir', default='prediction', type=str, help="the directory to save outputs to.")
    args = parser.parse_args()
    return args

def read_data_from_file(input_file):
    data = []
    rf = open(input_file, "r")
    for line in rf:
        example = json.loads(line)
        data.append(example)
    rf.close()
    return data

def checks_terminate_message(msg):
    if isinstance(msg, str):
        return msg.find("TERMINATE") > -1
    elif isinstance(msg, dict) and 'content' in msg:
        return msg['content'].find("TERMINATE") > -1
    else:
        print(type(msg), msg)
        raise NotImplementedError
    
def main():
    args = get_args()
    
    if args.input_file:
        input_examples = read_data_from_file(args.input_file)
    else:
        input_examples = load_dataset("zixianma/mnms", split="test_human_verified_filtered")

    if args.model.find('gpt') > -1:
        config_list = [{
            'model': args.model,
            'api_key': os.getenv("OPENAI_API_KEY"),
        }]
    elif args.model in ['meta-llama/Llama-2-70b-chat-hf', "google/gemma-7b-it", "mistralai/Mixtral-8x7B-Instruct-v0.1"]:
        config_list = [{
            "model": args.model,
            "base_url": "https://api.deepinfra.com/v1/openai",
            "api_key": os.getenv("DEEPINFRA_API_KEY")
        }]
    elif args.model.find('gemini') > -1:
        config_list = [{
            "model": args.model,
            "api_type": "google",
            "api_key": os.getenv("GOOGLE_API_KEY")
        }]
    else:
        config_list = [{
            "model": args.model,
            "base_url": "http://localhost:8000/v1",
            "api_key": "NULL"
        }]
    
    print(config_list)
    llm_config = {
        "seed": args.seed,
        "config_list": config_list,
    }

    # Set up output json file 
    run_id = f"{args.exp_id + '-' if args.exp_id else ''}{args.model}-{args.plan_mode}-{args.action_format}"
    run_id += f"{'-verify' if args.verify else ''}{'-execute' if args.execute else ''}-max-reply-{str(args.max_reply)}-seed-{str(args.seed)}"
    run_id = run_id.replace('/', '-')
    wf_name = f"{args.output_dir}/{run_id}.json"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Record ids of examples which we have run inference on (to skip these)
    has_inferenced = []
    if os.path.exists(wf_name):
        rf = open(wf_name, "r")
        for line in rf:
            data = json.loads(line)
            has_inferenced.append(data["id"])
        rf.close()
    wf = open(wf_name, "a")

    # Set up the planning agent with the correct prompt template and parser/verifier/executor modules
    feedback_generator = FeedbackPrompt(plan_mode=args.plan_mode, action_format=args.action_format)
    if args.plan_mode == "multi-step":
        if args.action_format == "json":
            prompt_generator = JsonGenPrompt() 
            parser =  Parser(plan_mode=args.plan_mode)
            verifier =  Verifier(tool_metadata=TOOL_METADATA) if args.verify else None 
            executor =  Executor(result_folder=os.path.join(RESULT_PATH, run_id)) if args.execute else None 
        elif args.action_format == "code":
            prompt_generator = CodeGenPrompt() 
            parser =  CodeParser(plan_mode=args.plan_mode)
            verifier =  CodeVerifier(tool_metadata=TOOL_METADATA) if args.verify else None 
            executor =  CodeExecutor(result_folder=os.path.join(RESULT_PATH, run_id)) if args.execute else None 
        else:
            raise NotImplementedError
        
        user = MnmsUserAgent(
            name="user_agent",
            human_input_mode='NEVER',
            max_consecutive_auto_reply=args.max_reply,
            is_termination_msg=checks_terminate_message,
            prompt_generator=prompt_generator,
            feedback_generator=feedback_generator,
            parser=parser,
            verifier=verifier,
            executor=executor
        )
    elif args.plan_mode == "step-by-step":
        if args.action_format == "json":
            prompt_generator = ReACTPrompt()
            feedback_generator = FeedbackPrompt(plan_mode=args.plan_mode, action_format="json")
            verifier = Verifier(tool_metadata=TOOL_METADATA) if args.verify else None
            parser = Parser(plan_mode=args.plan_mode)
            executor = Executor(log_obs=True, result_folder=os.path.join(RESULT_PATH, run_id)) if args.execute else None 
        else:
            # This implementation does not support step-by-step code generation
            raise NotImplementedError

        user = MnmsUserAgentLocal(
            name="user_agent",
            human_input_mode='NEVER',
            max_consecutive_auto_reply=args.max_reply,
            is_termination_msg=checks_terminate_message,
            prompt_generator=prompt_generator,
            feedback_generator=feedback_generator,
            parser=parser,
            verifier=verifier,
            executor=executor
        )
    else:
        raise NotImplementedError

    # Run the planning experiment
    all_messages = {}
    for idx, example in enumerate(input_examples):
        if example['id'] in has_inferenced:
            continue
        else:
            # Re-initialize the planner agent for each example to keep track of token usage for each run
            planner = AssistantAgent(
                name="planner",
                llm_config=llm_config,
                # The default system message of the AssistantAgent is overwritten here
                # system_message=system_message
            )
            query = example['user_request']
            print(query)
            
            try:
                user.initiate_chat(
                    planner,
                    message=query,
                    task_id=example['id'],
                    log_prompt_only=args.simulate
                )
                all_messages = planner.chat_messages
            except Exception as e:
                print(e)
                print(f"skipping {example['id']}..")
                all_messages = {'error': e.message if hasattr(e, 'message') else f"{e}"}

            if not args.simulate and args.output_dir:
                if 'error' in all_messages:
                    example['all_messages'] = all_messages
                else:
                    messages = {agent.name: msg for agent, msg in all_messages.items()}
                
                    # Exclude the initial prompt when saving the messages as it is very long
                    messages['user_agent'] = messages['user_agent'][1:]
                    example['all_messages'] = messages
       
                example['prediction'] = user.current_plan 
                example['feedback_types'] = user.feedback_types 
                example['usage_summary'] = {'total': planner.client.total_usage_summary, 'actual': planner.client.actual_usage_summary}
                
                save_keys = ['id', 'user_request', 'prediction', 'all_messages', 'usage_summary', 'feedback_types']
                output_dict = {k: example[k] for k in save_keys}
                wf.write(json.dumps(output_dict) + "\n")
                wf.flush()
            # print("FINAL usage:")
            # planner.client.print_usage_summary() 

            user.reset()
            planner.reset()
                    
if __name__ == '__main__':
    main()