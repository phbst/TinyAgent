from typing import Dict, List, Optional, Tuple, Union
import json5

from components.LLM import OpenAIModel
from components.Tools import Tools

# from LLM import OpenAIModel
# from Tools import Tools



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
            func_name=text[i+len('\nAction:'):j].strip()
            func_args=text[j+len('\nAction Input:'):k].strip()
            text=text[:k].strip()

        return func_name,func_args,text
    
    def call_plugin(self, func_name, func_args):
        
        func_args = json5.loads(func_args)

        if func_name=="google_search":

            return '\nObservation:' + self.tools.google_search(**func_args)
            
        
    def chat_by_func(self,query,history=[]):
        res_model_1=self.llm.chat(query,self.template_prompt)
        func_name,func_args,text=self.select_func_call(res_model_1)
        if func_name:
            res_func=self.call_plugin(func_name, func_args)
            res_model_2=text+res_func
        result=self.llm.chat(res_model_2,self.template_prompt)

        show='第一次模型交互：\n'+text+'\n\n'+f'执行{func_name}的结果：\n'+res_func+'\n\n'+'第二次交互：\n'+res_model_2+'\n\n'+'最终结果：\n'+result

        return  show


# test=Agent()
# a=test.llm.chat('你认识phb吗',test.template_prompt)
# print(a,'\n____________________')
# b,c,d=test.select_func_call(a)
# print(b,c,d,sep='\n')
# e=test.call_plugin(b,c)
# print(e,'\n____________________')
# f=d+e
# print(f,'\n____________________')
# g=test.llm.chat(f,test.template_prompt)
# print(g,'\n____________________')