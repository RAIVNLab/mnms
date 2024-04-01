import os
import pickle
from .config import *
from .utils import pipeline, save_result, save_output

class Executor:
    def __init__(
        self, 
        save_result: bool = True, 
        log_obs: bool = True, 
        result_folder: str = RESULT_PATH
    ) -> None:
        self.save_result = save_result
        self.log_obs = log_obs
        self.result_folder = result_folder

    def execute(self, task_id, task_nodes):
        """
        Example input:
        task_nodes = {'id': 0, 'name': 'image generation', 'args': {'text': 'an image depicting a woman worker looking at rear view mirror smiling'}
        or
        task_nodes = [{'id': 0, 'name': 'image generation', 'args': {'text': 'an image depicting a woman worker looking at rear view mirror smiling'}}, 
        {'id': 1, 'name': 'image captioning', 'args': {'image': '<node-0>.image'}}]   
        """
        try:
            if isinstance(task_nodes, dict):
                output =  pipeline(task_id, [task_nodes], result_folder=self.result_folder)
            elif isinstance(task_nodes, list):
                output =  pipeline(task_id, task_nodes, result_folder=self.result_folder)
            else:
                print("input nodes are invalid.")
                raise ValueError
            if self.save_result:
                save_result(task_id, output['path'], result_folder=self.result_folder)
            if self.log_obs:
                last_node_id = max(list(output['value'].keys()))
                return {'status': True, 'message': f"Execution succeeded. The output of ACTION {last_node_id} is {output['value'][last_node_id]}.\n"}
            else:
                return {'status': True, 'message': f"Execution succeeded."}
        except Exception as err:
            return {'status': False, 'message': f"Execution failed with {type(err)}: {err}."}



class CodeExecutor(Executor):
    def __init__(
        self, 
        save_result: bool = True, 
        log_obs: bool = True, 
        result_folder: str = RESULT_PATH
        ):
        super().__init__(save_result, log_obs, result_folder)

    def save_result_as_pickle(self, task_id, result):
        full_result_path = os.path.join(self.result_folder, str(task_id))
        output_dict_path_pickle = os.path.join(full_result_path, f"result.pkl")
        with open(output_dict_path_pickle, "wb") as file:
            pickle.dump(result, file)   

    def execute(self, task_id, code_snippet):
        """
        Example input:
        def solve():
            output0 = image_generation(text="an image depicting a woman worker looking at rear view mirror smiling")
            output1 = image_captioning(image=output0['image'])
            result = {0: output0, 1: output1}
            return result
        """
        try:
            # Avoid execution getting stuck waiting for inputs
            if code_snippet.find('input(') > -1 or code_snippet.find('stdin') > -1:
                return {'status': False, 'message': "Code snippet contains input() or stdin function, which is not supported."}
        
            import_lib = "import json\n"
            import_lib += "from mnms.execution.tool_api import *\n"""

            final_code = import_lib + code_snippet
            print(final_code)
            global_vars = {}
            exec(final_code, global_vars)
            result = global_vars['solve']() 
            outputs_path = {} 
            if isinstance(result, dict):
                for node_id, content in result.items():
                    if isinstance(content, dict):
                        output_path = save_output(task_id, node_id, content, self.result_folder)
                        outputs_path[str(node_id)] = output_path
                    else:
                        self.save_result_as_pickle(task_id, result)
                        break
                if len(outputs_path) > 0:
                    save_result(task_id, outputs_path, result_folder=self.result_folder)
            else:
                self.save_result_as_pickle(task_id, result)

            if self.log_obs:
                return {'status': True, 'message': f"Execution succeeded. The output is {result}.\n"}
            else:
                return {'status': True, 'message': f"Execution succeeded."}
        except Exception as err:
            return {'status': False, 'message': f"Execution failed with {type(err)}: {err}."}



        