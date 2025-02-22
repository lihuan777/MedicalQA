
# Secret Key：bce-v3/ALTAK-HrNgstJsTZuF7YBIRovZO/d24d7b095fe1321dc7bbcb9219837122d149bb3b
# API Key：04024eb16a1b45e1b7956994136f85b6


import requests
import json

API_KEY = "GsVH3kfWMBL9vSEtUwOzXkPi"
SECRET_KEY = "xtwB62o8xrV36eCEmeZssGXPNvrrVclw"


# https://cloud.baidu.com/doc/WENXINWORKSHOP/s/7lxwwtafj

import requests
import json

def get_access_token():
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """
        
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=GsVH3kfWMBL9vSEtUwOzXkPi&client_secret=xtwB62o8xrV36eCEmeZssGXPNvrrVclw"
    
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")

def main():
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-4.0-turbo-8k?access_token=" + get_access_token()
    
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": "介绍一下北京"
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    
    print(response.text)
    

if __name__ == '__main__':
    main()
