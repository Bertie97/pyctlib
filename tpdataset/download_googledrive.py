import requests
import hashlib

def download_file_from_google_drive(id, destination, md5=None):
    URL = "https://docs.google.com/uc?export=download"

    print("start download {} form google drive: {}".format(destination, URL + "&id=" + id))

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)
    if md5:
        assert hashlib.md5(open(destination, 'rb').read()).hexdigest() == md5

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    step = 1

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                step += 1
                f.write(chunk)
            if step % 100 == 0:
                print(destination, step)

if __name__ == "__main__":
    file_id = '0B7EVK8r0v71pY0NSMzRuSXJEVkk'
    destination = 'list_eval_partition.txt'
    download_file_from_google_drive(file_id, destination)
