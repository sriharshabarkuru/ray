# from configparse import ConfigParser
import configparser
import requests
config = configparser.ConfigParser()
import sys


# Check if a parameter is provided
if len(sys.argv) > 2:
    param1 = sys.argv[1]
    param2 = sys.argv[2]
    print("uuid:", param1)
    print("path:", param2)
else:
    print("Please provide uuid & image path.")
    exit(1)

# get http url & token
ini_file = "/tmp/.d3x.ini"
config.read(ini_file)
url = config.get("default","url")
token = config.get("default","auth-token")


# get deployment details
headers = {'Authorization': token}
r = requests.get(f"{url}/llm/api/deployments/{param1}", headers=headers, verify=False)
deployment = r.json()['deployment']

# get serving details
SERVING_TOKEN = deployment['serving_token']
SERVING_ENDPOINT = f"{url}{deployment['endpoint']}"
IMAGE_PATH =  param2 #"/home/sriharsha-barkuru/workspaces/default-workspace/ray/images/pull-over.png"

# convert image to bytes
with open(IMAGE_PATH, "rb") as image:
   f = image.read()
   image_bytes = bytearray(f)

# serving request
headers={'Authorization': SERVING_TOKEN}
resp = requests.post(SERVING_ENDPOINT, data=image_bytes, headers=headers, verify=False)
print (resp.json())
