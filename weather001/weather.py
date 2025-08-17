import requests
import json
from cities_db import get_city_coords
from openrouter_api import OpenRouterAPI

API_KEY = 'bb1bdc1d24213405ada30579a5a97f12'  
BASE_URL = 'http://api.openweathermap.org/data/2.5/weather'


def get_weather(city):
    try:
        coords = get_city_coords(city)
        if not coords:
            return "城市不在数据库中，请添加或使用其他城市"
            
        params = {
            'lat': coords['lat'],
            'lon': coords['lon'],
            'appid': API_KEY,
            'units': 'metric'
        }
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        
        data = response.json()

        print(data)
        
        weather_translation = {
            'clear sky': '晴天',
            'few clouds': '少云',
            'scattered clouds': '散云',
            'broken clouds': '多云',
            'overcast clouds': '阴天',
            'light rain': '小雨',
            'moderate rain': '中雨',
            'heavy intensity rain': '大雨',
            'very heavy rain': '暴雨',
            'extreme rain': '特大暴雨',
            'thunderstorm': '雷暴',
            'snow': '雪',
            'mist': '薄雾',
            'fog': '雾',
            'haze': '霾'
        }
        
        description = data['weather'][0]['description'].lower()
        translated_weather = weather_translation.get(description, data['weather'][0]['description'])
        
        return {
            '城市': data['name'],
            '温度': f"{data['main']['temp']}°C",
            '天气状况': translated_weather,
            '湿度': f"{data['main']['humidity']}%",
            '风速': f"{data['wind']['speed']} m/s"
        }
    except requests.exceptions.RequestException as e:
        return f"请求错误: {str(e)}"
    except (KeyError, json.JSONDecodeError):
        return "无效的API响应"


def test_weather_with_openrouter(openrouter_api, text):
    """
    使用OpenRouter API从文本中提取城市名称并查询天气
    :param openrouter_api: OpenRouterAPI实例
    :param text: 包含城市名称的文本
    :return: 天气信息或错误消息
    """
    try:
        # 使用大模型提取城市名称
        response = openrouter_api.query_weather(text)
        if "请求错误" in response or "无效的API响应" in response:
            return response
            
        # 从响应中提取城市名称
        city = text.split()[-1]  # 简单示例，实际应用中需要更复杂的NLP处理
        return get_weather(city)
    except Exception as e:
        return f"处理错误: {str(e)}"

if __name__ == '__main__':
    print("天气查询程序（输入q退出）")
    print("测试模式（输入t进入测试模式）")
    
    openrouter = OpenRouterAPI("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") if 'OpenRouterAPI' in globals() else None
    
    while True:
        user_input = input("\n请输入城市名称或包含城市名称的文本(输入q退出/t测试): ").strip()
        if user_input.lower() == 'q':
            break
        elif user_input.lower() == 't' and openrouter:
            test_text = input("请输入包含城市名称的测试文本: ")
            result = test_weather_with_openrouter(openrouter, test_text)
        else:
            result = get_weather(user_input)
        
        if isinstance(result, dict):
            print("\n当前天气信息：")
            for key, value in result.items():
                print(f"{key}: {value}")
        else:
            print(f"\n错误: {result}")