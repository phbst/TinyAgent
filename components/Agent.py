from typing import Dict, List, Optional, Tuple, Union
import json5

from components.LLM import OpenAIModel
from components.Tools import Tools




TOOL_DESC = """{english_name}: Call this tool to interact with the {chinese_name} API. What is the {chinese_name} API useful for? {description} Parameters: {parameters} Format the arguments as a JSON object."""

REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tools_description}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!


"""

class Agent:
    def __init__(self,path:str='') -> None:
        self.path=path
        self.tools=Tools()
        self.tools_config=self.tools.get_func()
        self.llm=OpenAIModel()
        self.template_prompt=self.build_template_prompt()

    def build_template_prompt(self):
        tools_description,tools_name=[],[]

        for tool in self.tools_config:
            tools_description.append(TOOL_DESC.format(**tool))
            tools_name.append(tool['english_name'])
        tools_description = '\n\n'.join(tools_description)
        tools_name = ','.join(tools_name)
            
        template_prompt=REACT_PROMPT.format(tools_description=tools_description,tools_name=tools_name)

        return template_prompt
    
    def select_func_call(self,text:str):
        func_name=''
        func_args=''
        i = text.rfind('\nAction:')
        j = text.rfind('\nAction Input:')
        k = text.rfind('\nObservation:')
        if j>i>=0:
            if k<0:
                text.strip()+'\nObservation:'
            k = text.rfind('\nObservation:')
            func_name=text[i+len('\nAction:'):j]
            func_args=text[j+len('\nAction Input:'):k]
            text=text[:k] 

        return func_name,func_args,text
    
    def call_plugin(self, func_name, func_args):
        func_args = json5.loads(func_args)
        if func_name == 'google_search':
            return '\nObservation:' + self.tools.google_search(**func_args)
        
    def chat_by_func(self,text,history=[]):
        res_model_1=self.llm.chat(text,self.template_prompt)
        func_name,func_args,text=self.select_func_call(res_model_1)
        if func_name:
            res_func=self.call_plugin(func_name, func_args)
            res_model_2=text+res_func
        result=self.llm.chat(res_model_2,self.template_prompt)
        return res_model_1+"\n-----------------------------\n" +res_model_2+"\n----------------------------\n"+result


