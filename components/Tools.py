import json 
import requests
import os 

class Tools:
    def __init__(self) -> None:
        self.tools_config=[
            {
                'chinese_name':'谷歌搜索',
                'english_name':'google_serach',
                'description':'谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。',
                'parameters':[
                    {
                        'name': 'search_query',
                        'description': '搜索关键词或短语',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ]
            },
        ]
        self.api=os.getenv('google_search_api')

    
    def get_func(self):
        return self.tools_config
    
    def google_serach(self,query:str):
        url = "https://google.serper.dev/search"

        payload = json.dumps({
        "q": query
        })
        headers = {
        'X-API-KEY': self.api,
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        return  response['organic'][0]['snippet']


