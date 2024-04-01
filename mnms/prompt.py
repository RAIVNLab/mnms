import json
from typing import Dict, List, Optional, Union
from mnms.constants import TOOL_METADATA, DEMO_EXAMPLES, REACT_DEMO_EXAMPLES

class PlanPrompt:
    def __init__(self, instruction, tool_metadata, demos, requirements):
        """Generate a planning prompt that consists of instruction, tool metadat, demos and requirements.

        Args:
            query: the query of the user
        Returns:
            str: the generated prompt.
        """
        self.instruction = instruction
        self.tool_metadata = tool_metadata
        self.demos = demos
        self.requirements = requirements

    def get_prompt_for_curr_query(self, query):
        """(Abstract method) Generate a prompt based on the received query.

        Args:
            query: the query of the user
        Returns:
            str: the generated prompt.
        """
        pass

class JsonGenPrompt(PlanPrompt):
    def __init__(self):
        tool_metadata = "# TOOL LIST #:\n"
        for tool, tool_meta in TOOL_METADATA.items():
            input_list = ', '.join([mod for mod in tool_meta['input']['arg_name']])
            output_list = ', '.join([mod for mod in tool_meta['output']['arg_name']])
            tool_metadata += tool + f": {tool_meta['description']} "
            tool_metadata += "Its input includes " + str(input_list) + ", and output includes " + str(output_list) + ".\n"
        instruction = """\n# GOAL #: Based on the above tools, I want you to generate the tool nodes to solve the # USER REQUEST #. """
        instruction += """Each tool node specifies the name and argument of a tool to call and must be in a strict JSON format, like: 
            {
                "nodes": [{
                    "id": an integer id of the tool, starting from 0, 
                    "name": "tool name must be from # TOOL LIST #", 
                    "args": { a dictionary of arguments for the tool. Either original text, or user-mentioned filename, or tag '<node-j>.text' (start from 0) to refer to the text output of the j-th node. }
                    }]
            } """ 
       
        demo_examples = DEMO_EXAMPLES
        demos = ""
        for demo in demo_examples:
            demo["result"] = {
                "nodes": demo["nodes"]
            }
            demos += f"""\n# EXAMPLE #:\n# USER REQUEST #: {demo["user_request"]}\n# RESULT #: {json.dumps(demo["result"])}"""
        requirements = """\n\n# REQUIREMENTS #:"""
        requirements += """\n1. The generated tool nodes can resolve the given user request # USER REQUEST # perfectly. Tool name must be selected from # TOOL LIST #;"""
        requirements += """\n2. The arguments of a tool must be the same number, modality, and format specified in # TOOL LIST #;"""
        requirements += """\n3. Use as few tools as possible.""" 
        super().__init__(instruction, tool_metadata, demos, requirements)

    def get_prompt_for_curr_query(self, query):
        request = f"""\n\n# USER REQUEST #: {query}\nNow please generate your result in a strict JSON format:\n# RESULT #:"""
        return  self.tool_metadata + self.instruction + self.requirements + self.demos + request
    


class ReACTPrompt(PlanPrompt):
    def __init__(self):
        tool_metadata = "# TOOL LIST #:\n"
        for tool, tool_meta in TOOL_METADATA.items():
            input_list = ', '.join([mod for mod in tool_meta['input']['arg_name']])
            output_list = ', '.join([mod for mod in tool_meta['output']['arg_name']])
            tool_metadata += tool + f": {tool_meta['description']} "
            tool_metadata += "Its input includes " + str(input_list) + ", and output includes " + str(output_list) + ".\n"
        instruction = """\n# GOAL #: Based on the above tools, I want you to reason about how to solve the # USER REQUEST # and generate the actions step by step. """
        instruction += """Each action specifies the name and arguments of a tool to call and must be in a strict JSON format, like: 
            {
                "id": an integer id of the tool, starting from 0, 
                "name": "tool name must be from # TOOL LIST #", 
                "args": { a dictionary of arguments for the tool. Either original text, or user-mentioned filename, or tag '<node-j>.text' (start from 0) to refer to the text output of the j-th node.}
            } """ 

        demo_examples = REACT_DEMO_EXAMPLES
        demos = ""
        for demo in demo_examples:
            demo["result"] = {
                "nodes": demo["nodes"]
            }
            result = ""
            for i, node in enumerate(demo['nodes']):
                result += f"""THOUGHT {i}: {"First" if i == 0 else f"Based on the user query and OBSERVATION {i-1}, then"}, I need to perform {node["name"]}.\n"""
                no_output_node = {"id": node['id'], "name": node['name'], "args": node['args']}
                # call the tool '{node["name"]}' with the arguments {json.dumps(node["args"])}
                result += f"""ACTION {i}: {json.dumps(no_output_node)}\n""" 
                if i < len(demo['nodes']) - 1:
                    obs_str = f"{node['output']}"
                    result += f"""OBSERVATION {i}: {obs_str}\n"""
            demos += f"""\n# EXAMPLE #:\n# USER REQUEST #: {demo["user_request"]}\n# RESULT #: \n{result}""" 
        requirements = """\n\n# REQUIREMENTS #:"""
        requirements += """\n1. The generated actions can resolve the given user request # USER REQUEST # perfectly. Tool name must be selected from # TOOL LIST #;"""
        requirements += """\n2. The arguments of a tool must be the same number, modality, and format specified in # TOOL LIST #;"""
        requirements += """\n3. Use as few tools as possible.""" 
        super().__init__(instruction, tool_metadata, demos, requirements)

    def get_prompt_for_curr_query(self, query):
        request = f"""\n# USER REQUEST #: {query}\nNow please generate only THOUGHT 0 and ACTION 0 in RESULT:\n# RESULT #:\n"""
        return  self.tool_metadata + self.instruction + self.requirements + self.demos + request



def format_python_str(input_str):
    """
    Turns "<node-0>.text" into "{output0['text']}" for proper formatting in python code
    """
    import re
    pattern = r"<node-\d+>\.[a-zA-Z]+"
   
    # Find all occurrences of the pattern in the text
    matches = re.findall(pattern, input_str)
    res_str = input_str
    for match in matches:
        # Extract node id from the string
        node_id = match[6]  
        start = match.find('.')
        last_node_out_arg_name = match[start+1:]
        
        res_str = res_str.replace(match, "{" + f"output{node_id}['{last_node_out_arg_name}']" + "}") 
    return res_str
    
def nodes_to_program(nodes):
    """
    Turns the json-format nodes into a python program named solve()
    Example:
    Input:
    [{"id": 0, "name": "image captioning", "args": {"image": "2327921.jpg"}}, 
    {"id": 1, "name": "text summarization", "args": {"text": output0["text"]}}]
    Output:
    def solve():
        output0 = image_captioning(image="2327921.jpg")
        output1 = text_summarization(text=output0['text'])
        result = {0: output0, 1: output1}
        return result
    """
    code = "def solve():\n"
    outputs = []
    for node in nodes:
        node_name = "output{}".format(node["id"])
        tool_name = node["name"].replace(" ", "_")
        args = []
        for k, v in node["args"].items():
            print(node["args"])
            arg_name = k
            v = v.replace('"', "'")
            
            if v.find("<node-") > -1:
                if k == "text":
                    arg_value = 'f"' + format_python_str(v) + '"'
                else:
                    input_node_id = v[6]
                    input_key = v[v.find('.') + 1:]
                    arg_value = f"output{input_node_id}['{input_key}']"
            else:
                arg_value = f'"{v}"'

            args.append(f"{arg_name}={arg_value}")

        arg_str = ", ".join(args)
        code += f"    {node_name} = {tool_name}({arg_str})\n"
        outputs.append(f'{node["id"]}: output{node["id"]}')

    final_output_str = 'result = {' + ', '.join(outputs) + '}'
    code += f'    {final_output_str}\n'
    code += '    return result'
    return code.strip()

class CodeGenPrompt(PlanPrompt):
    def __init__(self):
        tool_metadata = "# TOOL LIST #:\n"
        for tool, tool_meta in TOOL_METADATA.items():
            input_list = ', '.join([mod for mod in tool_meta['input']['arg_name']])
            output_list = ', '.join([mod for mod in tool_meta['output']['arg_name']])
            tool_func_str = tool.replace(' ', '_') + f'({input_list}) -> {output_list}: {tool_meta["description"]}\n'
            tool_metadata += tool_func_str

        instruction = """\n# GOAL #: Based on the above tools, I want you to generate a python program to solve the # USER REQUEST #."""
       
        demo_examples = DEMO_EXAMPLES
        demos = ""
        for demo in demo_examples:
            program = """```python\n"""
            program += nodes_to_program(demo['nodes'])
            program += """\n```"""
            demos += f"""\n# EXAMPLE #:\n# USER REQUEST #: {demo["user_request"]}\n# RESULT #: {program}"""
        requirements = """\n\n# REQUIREMENTS #:"""
        requirements += """\n1. The generated program can resolve the given user request # USER REQUEST # perfectly. The functions must be selected from # TOOL LIST #; """
        requirements += """\n2. The arguments of a function must be the same number, modality, and format specified in # TOOL LIST #;"""
        requirements += """\n3. Use as few tools as possible."""
        super().__init__(instruction, tool_metadata, demos, requirements)

    def get_prompt_for_curr_query(self, query):
        request = f"""\n\n# USER REQUEST #: {query}\nNow please generate your program enclosed in ```python ```:\n# RESULT #:"""
        return  self.tool_metadata + self.instruction + self.requirements + self.demos + request


class FeedbackPrompt:
    def __init__(self, plan_mode, action_format):
        self.plan_mode = plan_mode
        self.action_format = action_format
        if self.plan_mode == "step-by-step":
            self.keyword = "ACTION" 
        else: 
            if self.action_format == "code":
                self.keyword = "program enclosed in ```python ``` in # RESULT #"
            else:
                self.keyword = "# RESULT #"
        
        self.default_req_msg = f"\nPlease try generating the {self.keyword} again to fix the error. Or, reply with TERMINATE only if you believe this error is not fixable."
        self.msg_prefix = "OBSERVATION: " if self.plan_mode == "step-by-step" else ""

    def get_prompt(self, stage, error_status, error_msg, error_code='unknown'):
        if error_status:
            if self.plan_mode == "step-by-step":
                return self.msg_prefix + error_msg + "\nIf the request has been fulfilled, please reply with TERMINATE, otherwise please generate the next THOUGHT and ACTION."
            elif self.plan_mode == "multi-step":
                return self.msg_prefix + error_msg
            else:
                raise NotImplementedError(f"{self.plan_mode} has not been implemented.") 
        else:
            if stage == "parsing":
                if error_code == "json":
                    req_msg = f"\nPlease format the {self.keyword} to a strict JSON format " + "that starts with left curly bracket and ends with right curly bracket."
                    req_msg += """\nRequirements:\n1. Do not change the information in nodes;\n2. Consider changing double quotes to single quotes or vice versa where applicable;\n3. Consider removing extra curly brackets if any;\n4. Don't tolerate any possible irregular formatting to ensure that the generated content can be converted by json.loads()."""
                else: # unknown, or any other error codes
                    req_msg = self.default_req_msg
                return self.msg_prefix + error_msg + req_msg
            elif stage == "verification":
                if error_code == 'unknown':
                    return self.msg_prefix + error_msg + self.default_req_msg
                else:
                    return self.msg_prefix + error_msg + f"\nPlease try again and fix the {error_code} in the {self.keyword} while keeping other parts the same."
            elif stage == "execution":
                return self.msg_prefix + error_msg + self.default_req_msg
           
                

