import requests

url = "http://127.0.0.1:8000/processPdf"
files = {"file": opem("saple.pdf","rb")}

response = requests.post(url, files=files)
print(response.json())