import json

with open('train_json/videochatgpt_tune_.json', 'r') as f:
    train = json.load(f)

print(len(train))
print(train[0])