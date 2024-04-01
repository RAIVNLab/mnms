import networkx as nx
import pickle
import ast
import re
from mnms.constants import *

# the regular expression pattern to find the <node-x>.xxxx in the args
def find_node_patterns(text):
    # Regular expression pattern to find <node-x>.xxxx
    pattern = r"<node-\d+>\.[a-zA-Z]+"

    # Find all occurrences of the pattern in the text
    matches = re.findall(pattern, text)
    return matches

class Verifier:
    def __init__(self, tool_metadata):
        self.tools_dict = tool_metadata
        file = open("mnms/tool_graph.pkl",'rb')
        self.tool_graph = pickle.load(file)

    def create_graph_from_nodes(self, tool_nodes):
        G = nx.DiGraph()

        # Add nodes and edges based on the 'id' and 'args'
        for node in tool_nodes:
            node_id = node['id']
            G.add_node(node_id, name=node['name'], args=node['args'])

        for node in tool_nodes:
            # Check if there are any references to previous nodes in the args
            this_id = node['id']
            for arg_key, arg_value in node['args'].items():
                
                if isinstance(arg_value, str) and arg_value.startswith('<node-'):
                    # Extract the referenced node id
                    referenced_node_id = int(arg_value[6])
                    # Add an edge from the referenced node to the current node
                    G.add_edge(referenced_node_id, this_id)
        return G

    def verify_tool_name(self, node_name):
        status = True
        msgs = []
        err_codes = []
        if node_name not in self.tool_graph.nodes():
            status = False
            msg = f"{node_name} does not exist in the tool list."
            msgs.append(msg)
            err_codes.append("tools")
        output = {"status": status, "message": msgs, "error_code": err_codes}
        return output

    def verify_tool_args(self, node_name, node_args):
        status = True
        msgs = []
        err_codes = []
        # check arguments number
        input_args = self.tools_dict[node_name]['input']['arg_name']
        if len(input_args) != len(node_args):
            status = False
            msg = f"for {node_name}, {len(input_args)} arguments are expected, but there are {len(node_args)}."
            msgs.append(msg)
            err_codes.append(f"{node_name}'s arguments")
        # check for any missing arguments
        for arg in input_args:
            if arg not in node_args:
                status = False
                msg = f"'{arg}' is missing in the arguments of {node_name}."
                msgs.append(msg)
                err_codes.append(f"{node_name}'s arguments")
        # check for extra arguments
        for arg_name, arg_value in node_args.items():
            if arg_name not in input_args:
                status = False
                msg = f"'{arg_name}' should not appear in the arguments of {node_name}."
                msgs.append(msg)
                err_codes.append(f"{node_name}'s arguments")
        output = {"status": status, "message": msgs, "error_code": err_codes}
        return output

    def verify_tool_pair(self, prev_tool, curr_tool):
        status = True
        msgs = []
        err_codes = []
        # check edge existence
        if (prev_tool, curr_tool) not in self.tool_graph.edges():
            status = False
            msg = f"{curr_tool} should not follow {prev_tool}, " 
            reason = f"because it does not make sense to feed {self.tools_dict[prev_tool]['output']['desc']} into {curr_tool}, which expects the input to be {self.tools_dict[curr_tool]['input']['desc']}."
            msgs.append(msg + reason)
            err_codes.append("tools")
        output = {"status": status, "message": msgs, "error_code": err_codes}
        return output

    def verify_single_node(self, node):
        '''
        Verifies a single tool node by checking if the tool exists & if the tool's arguments follow the format correctly
        '''
        statuses = []
        msgs = []
        err_codes = []
        try:
            node_name = node['name']
            # check tool existence 
            output = self.verify_tool_name(node_name)
            status = output['status']
            statuses.append(status)
            if status:
                node_args = node['args']
                # check tool arguments
                output = self.verify_tool_args(node_name, node_args)
                statuses.append(output['status'])
                err_codes += output['error_code']
                msgs += output['message']
            else:
                err_codes += output['error_code']
                msgs += output['message']

            unique_err_codes = set(err_codes)
            unique_msgs = set(msgs)
            final_status = False not in statuses
            return {'status': final_status, 'message': 'Verification succeeded.' if final_status else  ' '.join(unique_msgs), 'error_code': ', '.join(unique_err_codes)}
        except Exception as err:
            return {'status': False, 'message': f"Unexpected {type(err)}: {err}.", 'error_code': 'unknown'}

    def verify(self, content):
        '''
        Verifies the content (i.e. a list of nodes) by checking a set of rules for each of the tool nodes and connections between nodes
        Example:
        [{'id': 0, 'name': 'visual question answering', 'args': {'image': '2415263.jpg', 'question': 'Who wears a shirt?'}}, 
        {'id': 1, 'name': 'text generation', 'args': {'text': 'a possible storyline following the scene of <node-0>.text'}}, 
        {'id': 2, 'name': 'text summarization', 'args': {'text': '<node-1>.text'}}]
        '''
        statuses = []
        msgs = []
        err_codes = []
        try: 
            graph = self.create_graph_from_nodes(content)
            for node in graph.nodes():
                node_name = graph.nodes[node]['name']
                # check tool existence 
                output = self.verify_tool_name(node_name)
                status = output['status']
                statuses.append(status)

                if status:
                    node_args = graph.nodes[node]['args']
                    # check tool args validity
                    output = self.verify_tool_args(node_name, node_args)
                    statuses.append(output['status'])
                    err_codes += output['error_code']
                    msgs += output['message']

                    # check for output format & reference inconsistency
                    for arg_name, arg_value in node_args.items():
                        if isinstance(arg_value, str) and arg_value.find('<node-') > -1:
                            patterns = find_node_patterns(arg_value)
                            if len(patterns) == 0:
                                statuses.append(False)
                                msg = f"{arg_value} has the wrong format. It should refer to a specific output of the last node i by <node-i>.key."
                                msgs.append(msg)
                                err_codes.append(f"{node_name}'s arguments")
                            for pattern in patterns:
                                last_node_id = int(pattern[6])
                                last_node_name = graph.nodes[last_node_id]['name']
                                output_mods = self.tools_dict[last_node_name]['output']['arg_name']
                                
                                start = pattern.find('.')
                                last_node_output_mod = pattern[start+1:]
                                if last_node_output_mod not in output_mods:
                                    statuses.append(False)
                                    msg = f"{last_node_output_mod} is not in the outputs of {last_node_name}, which outputs {str(output_mods)}."
                                    msgs.append(msg)
                                    err_codes.append(f"{node_name}'s arguments")             
                else:
                    err_codes += output['error_code']
                    msgs += output['message']
            # check edge existence
            node_names = nx.get_node_attributes(graph, 'name')
            for edge in graph.edges():
                src_name = node_names[edge[0]]
                tgt_name = node_names[edge[1]]

                if src_name in self.tool_graph.nodes() and tgt_name in self.tool_graph.nodes():
                    output = self.verify_tool_pair(src_name, tgt_name)
                    status = output['status']
                    statuses.append(status)
                    err_codes += output['error_code']
                    msgs += output['message']
                    
            unique_err_codes = set(err_codes)
            unique_msgs = set(msgs)
            final_status = False not in statuses
            return {'status': final_status, 'message': 'Verification succeeded.' if final_status else  ' '.join(unique_msgs), 'error_code': ', '.join(unique_err_codes)}
        except Exception as err:
            return {'status': False, 'message': f"Unexpected {type(err)}: {err}.", 'error_code': 'unknown'}


class CodeVerifier(Verifier):
    def __init__(self, tool_metadata):
        super().__init__(tool_metadata=tool_metadata)

    def get_tool_name_from_func_node(self, node):
        if hasattr(node.func, 'id'): 
            func_name = node.func.id
            tool_name = func_name.replace('_', ' ')
            return tool_name
        return None

    def verify(self, content):
        '''
        Example input:
        def solve():
            output0 = image_classification(image="10084.jpg")
            output1 = image_generation(text=f"a new, more detailed or stylized image of {output0['text']}")
            result = {0: output0, 1: output1}
            return result
        '''
        statuses = []
        msgs = []
        err_codes = []
        parsed_code = ast.parse(content)
        var_values = {}
        valid_tool_names_and_args = {}

        try:
            for node in ast.walk(parsed_code):
                # get var name to value mapping
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id :
                            var_values[target.id] = node.value
                # Verify the name and arguments of each tool
                if isinstance(node, ast.Call):
                    tool_name = self.get_tool_name_from_func_node(node)
                    output = self.verify_tool_name(tool_name)
                    status = output['status']
                    statuses.append(status)

                    if status:
                        tool_args = {}
                        # assuming the tool takes in a list of keyword arguments
                        for kwarg in node.keywords: 
                            var_name = kwarg.arg
                            if isinstance(kwarg.value, ast.Constant): 
                                # extract .value only if it's a constant
                                tool_args[var_name] = kwarg.value.value
                            else:
                                # otherwise keep the orginal ast object
                                tool_args[var_name] = kwarg.value
                                
                        # print(tool_args)
                        valid_tool_names_and_args[tool_name] = tool_args
                        # verify this tool's arguments
                        output = self.verify_tool_args(tool_name, tool_args)
                        statuses.append(output['status'])
                        err_codes += output['error_code']
                        msgs += output['message']
                    else:
                        err_codes += output['error_code']
                        msgs += output['message']
            
            print(valid_tool_names_and_args)
            # Verify edges and output reference
            # valid_tool_names_and_args = {'image classification': {'image': '10084.jpg'}, 'image generation': {'text': <ast.JoinedStr object at 0x7f602cb370a0>}}
            for tool_name, tool_args in valid_tool_names_and_args.items():
                for k, v in tool_args.items():
                    last_tool_names = []
                    last_node_out_keys = []

                    # We check for the following three common cases where a tool's input refers to last tool's output
                    if isinstance(v, ast.Name): # v = output0 (missing reference, will fail verification)
                        var_name = v.id
                        last_node = var_values[var_name] 
                        if isinstance(last_node, ast.Call):
                            last_tool_names = [self.get_tool_name_from_func_node(last_node)]
                            last_node_out_keys = [None] 
                    elif isinstance(v, ast.Subscript): # v = output0["text"]
                        var_name = v.value.id #  output0["text"] -> output0
                        last_node = var_values[var_name] 
                        if isinstance(last_node, ast.Call):
                            last_tool_names = [self.get_tool_name_from_func_node(last_node)]
                            last_node_out_keys = [v.slice.value] # output0["text"] -> ["text"]
                    elif isinstance(v, ast.JoinedStr): # v = f"a new, more detailed or stylized image of {output0['text']}"
                        subscripts = [val.value for val in v.values if isinstance(val, ast.FormattedValue)] # [<ast.Subscript object at 0x7f966d411b20>] i.e. output0['text']
                        last_outs = [val.value.id for val in subscripts] # ["output0"]
                        last_nodes = [var_values[out_name] for out_name in last_outs] # [<ast.Call object at 0x7f1b14580cd0>]
                        last_tool_names = [self.get_tool_name_from_func_node(last_node) if isinstance(last_node, ast.Call) else None for last_node in last_nodes] # ["image classification"]
                        last_node_out_keys = [subscript.slice.value for subscript in subscripts] # ["text"]
                    
                    for i, last_tool_name in enumerate(last_tool_names):
                        if last_tool_name:
                            if last_tool_name in self.tool_graph.nodes() and tool_name in self.tool_graph.nodes():
                                # check for edge existence between last tool and this current tool
                                output = self.verify_tool_pair(last_tool_name, tool_name)
                                statuses.append(output['status'])
                                err_codes += output['error_code']
                                msgs += output['message']

                                output_keys = self.tools_dict[last_tool_name]['output']['arg_name']
                                last_node_out_key = last_node_out_keys[i]
                                if last_node_out_key:
                                    if last_node_out_key not in output_keys:
                                        status = False
                                        statuses.append(status)
                                        msg = f"{last_node_out_key} is not in the outputs of {last_tool_name}, which outputs {str(output_keys)}."
                                        msgs.append(msg)
                                        err_codes.append(f"{tool_name}'s arguments")
                                else:
                                    statuses.append(False)
                                    msg = f"{tool_name}'s argument has the wrong format. It should refer to a specific output of the last node by output['key'] where key can be one of {str(output_keys)}."
                                    msgs.append(msg)
                                    err_codes.append(f"{tool_name}'s arguments")                          
            # print(statuses)
            final_status = False not in statuses
            unique_err_codes = set(err_codes)
            unique_msgs = set(msgs)
            return {'status': final_status, 'message': 'Verification succeeded.' if final_status else  ' '.join(unique_msgs), 'error_code': ', '.join(unique_err_codes)}
        except Exception as err:
            return {'status': False, 'message': f"Unexpected {type(err)}: {err}.", 'error_code': 'unknown'}
        
# def main():
    
#     verifier = Verifier(tool_metadata=TOOL_METADATA)
#     nodes = [{'id': 0, 'name': 'visual question answering', 'args': {'image': '2415263.jpg', 'question': 'Who wears a shirt?'}}, {'id': 1, 'name': 'text generation', 'args': {'text': 'a possible storyline following the scene of <node-0>.text'}}, {'id': 2, 'name': 'text summarization', 'args': {'text': '<node-1>.text'}}]
#     results = verifier.verify(nodes)
#     print(nodes)

#     verifier = CodeVerifier(tool_metadata=TOOL_METADATA)
#     program = """def solve():\n"""
#     program += """    output0 = image_classification(image="10084.jpg")\n"""
#     program += """    output1 = image_generation(text=f"a new, more detailed or stylized image that matches the classification of {output0['text']}")\n"""
#     # program += """    output1 = text_generation(text="generate a prefix")\n"""
#     # program += """    output2 = image_generation(text=f"{output1['text']} of {output0['text']}")\n"""
#     program += """    result = {0: output0, 1: output1}\n"""
#     program += """    return result""" 
#     print(program)
#     results = verifier.verify(program)

#     print(results)


# if __name__ == '__main__':
#     main()