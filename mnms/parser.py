import json

class Parser:
    def __init__(self, plan_mode):
        self.plan_mode = plan_mode

    def parse(self, response):
        if isinstance(response, dict) and 'content' in response:
            response = response['content']
       
        oring_content = response.replace("\n", "")
        oring_content = oring_content.replace("\_", "_")
        content = oring_content.replace("\\", "")
        
        try:
            start_pos = content.find("RESULT #:")
            if start_pos != -1:
                content = content[start_pos+len("RESULT #:"):]

            start_pos = content.find("{")
            if start_pos != -1:
                content = content[start_pos:]

            end_pos = content.rfind("}")
            if end_pos != -1:
                content = content[:end_pos+1]

            content = json.loads(content)
            return {'status': True, 'content': content['nodes'] if (self.plan_mode == 'multi-step' and 'nodes' in content) else content, 'message': 'Parsing succeeded.', 'error_code': ''}
        except json.JSONDecodeError as err:
            return {'status': False, 'content': content, 'message': f"{type(err)}: {err}.", 'error_code': 'json'} 
        except Exception as err:
            return {'status': False, 'content': content, 'message': f"Unexpected {type(err)}: {err}.", 'error_code': 'unknown'}



class CodeParser(Parser):
    def __init__(self, plan_mode):
        super().__init__(plan_mode)

    def parse(self, response):
        if isinstance(response, dict) and 'content' in response:
            response = response['content']
        oring_content = response.replace("\_", "_")
        content = oring_content.replace("\\", "")
        
        try:
            start_pos = content.find("RESULT #:")
            if start_pos != -1:
                content = content[start_pos+len("RESULT #:"):]
            
            start_pos = content.find("```python")
            if start_pos != -1:
                content = content[start_pos+len("```python"):]

            end_pos = content.find("```")
            if end_pos != -1:
                content = content[:end_pos]
            
            if start_pos == -1 or end_pos == -1:
                return {'status': False, 'content': content, 'message': 'Program is NOT enclosed in ```python``` properly.', 'error_code': 'unknown'}
            if len(content) > 0:
                compile(content, "prog.py", "exec")
                return {'status': True, 'content': content, 'message': 'Parsing succeeded.', 'error_code': ''}
            else:
                return {'status': False, 'content': content, 'message': "The content is empty, or it failed to parse the content correctly.", 'error_code': 'unknown'}
        except Exception as err:
            return {'status': False, 'content': content, 'message': f"Unexpected {type(err)}: {err}.", 'error_code': 'unknown'}


def main():
    parser = CodeParser()
    program = """RESULT #:```python\n"""
    program += """def solve():\n"""
    program += """    output0 = text_generation(prompt="Would you rather have an Apple Watch - or a BABY?")\n"""  
    program += """    output1 = text_summarization(text=output0["text"])\n"""
    program += """    return output1\n""" 
    program += """```"""
    print(program)
    results = parser.parse(program)
    print(results)

    parser = Parser()
    # msg = """# RESULT #:\n\nTHOUGHT 0: First, I need to search for the movie 'The Shape of Water' from 2017 to gather information.\nACTION 0: {\"id\": 0, \"name\": \"search movie\", \"args\": {\"movie_title\": \"The Shape of Water\", \"movie_year\": 2017}} \n\n# Now you can execute the above action to proceed."""
    msg = """"# RESULT #:\n\n{\n  \"nodes\": [\n    {\n      \"id\": 0,\n      \"name\": \"get date fact\",\n      \"args\": {\n        \"date\": \"May 7\"\n      }\n    }\n  ]\n}"""
    print(msg)
    results = parser.parse(msg)
    print(results)

if __name__ == '__main__':
    main()