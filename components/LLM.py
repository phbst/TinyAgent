from typing import Dict, List, Optional, Tuple, Union

import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_openai import OpenAI,ChatOpenAI
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage

class BaseModel:
    def __init__(self,path:str='') -> None:
        if path!='':
            self.tokenizer=AutoTokenizer.from_pretrained(path)
            self.model=AutoModelForCausalLM.from_pretrained(path)
        elif path=='':
            self.model= ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
            
    
    def chat(self,query:str, prompt_template: str='', history:List=[]):
        pass

    def load_model(self):
        pass

class OpenAIModel(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)

    def chat(self,query:str, prompt_template: str='', history:List=[]):
        message=[]
        message.append(HumanMessage(content=prompt_template))
        message.append(HumanMessage(content=query))
        # message=prompt_template+f'\nQuestion: {query}'
        # message=HumanMessage(content=message)
        res=self.model.invoke(message)
        return res.content
