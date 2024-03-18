import requests

def call_app(prompt):
    url = "http://127.0.0.1:5000/"
    data = {"text": prompt}
    response = requests.post(url, json=data)
    print(response.json()['text'])


if __name__ == "__main__":
    for i in range(3):
        prompt = input("Enter text: ")
        call_app(prompt)    