import requests
import json

class OpenRouterAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
    
    def query_weather(self, city_name):
        """
        查询指定城市的天气信息
        :param city_name: 城市名称
        :return: 包含天气信息的字典或错误消息字符串
        """
        try:
            # 构造OpenRouter API请求
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek/deepseek-chat-v3-0324:free",
                "messages": [
                    {
                        "role": "user",
                        "content": f"告诉我{city_name}的当前天气情况，包括温度、天气状况、湿度和风速"
                    }
                ]
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            print(data)
            return data['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            return f"请求错误: {str(e)}"
        except (KeyError, json.JSONDecodeError):
            return "无效的API响应"