# TinyAgent（开发中，预计5天）

在`ChatGPT`横空出世，夺走`Bert`的桂冠之后，大模型愈发的火热，国内各种模型层出不穷，史称“百模大战”。大模型的能力是毋庸置疑的，但大模型在一些实时的问题上，或是某些专有领域的问题上，可能会显得有些力不从心。因此，我们需要一些工具来为大模型赋能，给大模型一个抓手，让大模型和现实世界发生的事情对齐颗粒度，这样我们就获得了一个更好的用的大模型。



一步一步手写`Agent`，可以让我们对`Agent`的构成和运作更加的了解。以下是`React`论文中一些小例子。

> 论文：***[ReAct: Synergizing Reasoning and Acting in Language Models](http://arxiv.org/abs/2210.03629)***



## 项目结构

```
tinyAgent
├─ build.ipynb
├─ components
│  ├─ Agent.py
│  ├─ LLM.py
│  └─ Tools.py
├─ README.md
├─ static
└─ test.ipynb

```




## 实现细节

### Step 1: 构造大模型（LLM.py）

考虑到显卡资源有限，演示直接直接使用Langchain 的OpenAI API来构建LLM模块，当然你也可以选择加载本地的大模型。



首先构建一个BaseModel类，可以选择性传入一个path，表示你本地的LLM.

默认为空字符串，不传则代表使用API.

```python
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
            
    
    def chat(self, prompt: str, history: List[dict]):
        pass

    def load_model(self):
        pass

```

构造一个子类OpenAIModel，这个类是我们使用API的类，继承于BaseModel。

如果你使用本地的大模型，需要自行构造子类。

```python
class OpenAIModel(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)

    def chat(self,query:str, prompt_template: str='', history:List=[]):

        message=prompt_template+f'\n\n{query}'
        message=HumanMessage(content=message)

        res=self.model.invoke(message)
        return res
```



Step 2: 构造方法类（Tools.py）

```python
pass
```



Step 3: 构造Agent类（Agent.py）



```python
pass
```



Step 4：Agent,启动！



```python
pass
```



## 论文参考

- [ReAct: Synergizing Reasoning and Acting in Language Models](http://arxiv.org/abs/2210.03629)