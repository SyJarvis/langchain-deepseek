from pydantic import BaseModel, Field
import os
import requests
import json
from .exceptions import AuthenticationError, BadRequestError, APITimeoutError

DEFAULT_BASE_URL = "https://api.deepseek.com"


class RestAPI:
    def __init__(
        self,
        base_url: str = None,
        api_key: str = None
    ):
        env_base_url = os.getenv("DEEPSEEK_BASE_URL")
        env_api_key = os.getenv("DEEPSEEK_API_KEY")
        if base_url is None or not isinstance(base_url, str):
            base_url = env_base_url if env_base_url else DEFAULT_BASE_URL
        if api_key is None:
            if env_api_key is None:
                raise ValueError("`DEEPSEEK_API_KEY` is required")
            else:
                api_key = env_api_key
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()

    def action_post(self, request_path: str, datas: str=None, **kwargs):
        """POST"""
        url = self.base_url + "/" + request_path
        headers = self._generate_headers()
        stream = True if kwargs["stream"] == True else False
        payload = datas if datas else json.dumps(kwargs)
        for _ in range(3):
            try:
                response = self.session.post(url, headers=headers, data=payload)
            except Exception as e:
                print("Retry for HTTP Error ...")
                continue
            else:
                if response.status_code == 401:
                    raise AuthenticationError(
                        "Authentication failure: invalid API key or unauthorized access",
                        response=response,
                        body=response.text
                    )
                elif response.status_code == 400:
                    raise BadRequestError(
                        "Request parameter error",
                        response=response,
                        body=response.text
                    )
                elif response.status_code == 200:
                    break
        else:
            raise APITimeoutError("Connection timeout")

        if stream:
            return response
        else:
            if response.text:
                resp = response.json()
                return resp
            else:
                return {}
        
    def close(self):
        """关闭会话"""
        self.session.close()

    def _generate_headers(self, token=None) -> dict:
        if token is None:
            token = self.api_key
        return {
            "Content-Type": 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {token}'
        }