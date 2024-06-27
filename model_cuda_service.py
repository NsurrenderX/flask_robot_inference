import os
import cv2
import torch
import json
import time
import mii
import base64
import requests
import transformers
import numpy as np

from configs import H4ArgumentParser, DataArguments, VLAModelArguments, TATSModelArguments
from tokenizer import VQGANVisionActionEval, VideoData, get_image_action_dataloader, count_parameters
from torchvision import transforms
from PIL import Image
from flask import Flask, jsonify, request

def decode_b64_image(b64image):
    # Decode the base64 image string
    str_decode = base64.b64decode(b64image)
    np_image = np.frombuffer(str_decode, np.uint8)
    image_cv2 = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    return image_cv2

@torch.no_grad()
def encode(instance_data, model, tats_args, device):

    transform = transforms.Compose([
        transforms.Resize((tats_args.resolution, tats_args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)) # To [-0.5, 0.5]
    ])
    video = []
    for i in range(6):
        for img in instance_data['images']:
            img = torch.from_numpy(img).permute(2,0,1)
            img = transforms.ToPILImage()(img)
            img = transform(img)
            video.append(img)
    video = torch.stack(video).permute(1,0,2,3).to(device) # [C, T, H, W]
    action = torch.tensor(instance_data['actions']).to(device) # [T, 7]

    # normalize the actions
    action = (action - torch.tensor(instance_data['mean']).to(device)) / torch.tensor(instance_data['std']).to(device)

    _, _, vq_output, vq_output_action = model(video.unsqueeze(0), action.unsqueeze(0))
    video_tokens, action_tokens = vq_output['encodings'].reshape(-1), vq_output_action['encodings'].reshape(-1) # video tokens: 3*256=768, action tokens: 6*7=42

    return video_tokens, action_tokens

@torch.no_grad()
def call_vla(instance_data: dict,
             video_tokens: torch.Tensor, action_tokens: torch.Tensor, 
             vla_pipe: mii.pipeline, data_args: DataArguments, device):

    video_tokens = video_tokens.cpu().numpy().tolist()
    action_tokens = action_tokens.cpu().numpy().tolist()

    input_text = '<bott_i>' + instance_data['task_description'] + '<eott_i>' + \
                '<bots_i>' + instance_data['scene_description'] + '<eots_i>' + \
                '<botp_i>' + instance_data['clip_description'] + '<eotp_i>'

    if data_args.action_before_vision:
        input_text += '<boa_i>' + ''.join([f'<va{str(x)}>' for x in action_tokens]) + '<eoa_i>' + \
                '<bov_i>' + ''.join([f'<va{str(x)}>' for x in video_tokens]) + '<eov_i>'
    else:
        input_text += '<bov_i>' + ''.join([f'<va{str(x)}>' for x in video_tokens]) + '<eov_i>' + \
                '<boa_i>' + ''.join([f'<va{str(x)}>' for x in action_tokens]) + '<eoa_i>'

    output = vla_pipe([input_text], max_new_tokens=1024)
    
    output_text = output[0].generated_text
    with open('service.log', 'a') as f:
        f.write(output_text)

    # print(output_text)
    for x in output_text.split('<eoa_o>')[0].split('<boa_o>')[-1].split('<va'):
        if x != '':
            print(x)
            print("\n\n")
    output_action_tokens_pred = [int(x[:-1]) for x in output_text.split('<eoa_o>')[0].split('<boa_o>')[-1].split('<va') if x != '']
    output_action_tokens_pred = torch.tensor(output_action_tokens_pred, device=device).unsqueeze(0).reshape(1, 6, 7)

    output_clip_description_pred = output_text.split('<eotp_o>')[0].split('<botp_o>')[-1]

    return output_action_tokens_pred, output_clip_description_pred

def call_models(instance_data, model_vq: VQGANVisionActionEval, vla_pipe: mii.pipeline, 
                tats_args: TATSModelArguments, data_args: DataArguments, device):
    
    video_tokens, action_tokens = encode(instance_data, model_vq, tats_args, device=device)

    output_action_tokens_pred, output_clip_description_pred = call_vla(instance_data, video_tokens, action_tokens, vla_pipe, data_args, device)

    output_action_pred = model_vq.decode_action(output_action_tokens_pred).squeeze(0).detach().cpu() # 6, 7

    instance_data['clip_description'] = output_clip_description_pred
    instance_data['actions'] = output_action_pred.tolist()

    return instance_data

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Cuda realtime vla model inference service!'

@app.route('/predict', methods=['POST'])
def predict():
    global model_vq, vla_pipe, device, tats_args, data_args

    resp = request.get_json()
    images = resp['images']
    time_a = time.time()
    instance_data = {}
    instance_data["images"] = []
    image_real_1 = decode_b64_image(images['real_1'])
    # instance_data['images'].append(image_real_1)
    image_real_2 = decode_b64_image(images['real_2'])
    # instance_data['images'].append(image_real_2)
    image_k4a_1 = decode_b64_image(images['k4a_0'])
    instance_data['images'].append(image_k4a_1)
    image_k4a_2 = decode_b64_image(images['k4a_1'])
    # instance_data['images'].append(image_k4a_2)

    instance_data['task_description'] = resp['task_description']
    instance_data['clip_description'] = resp['clip_description']
    instance_data['scene_description'] = resp['scene_description']
    instance_data['mean'] = resp['mean']
    instance_data['std'] = resp['std']

    if resp['first_action']:
        to_append = [0,0,0,0,0,0,0]
        instance_data['actions'] = []
        for i in range(6):
            instance_data['actions'].append(to_append)
    else:
        instance_data['actions'] = resp['actions']
    
    instance_data = call_models(instance_data, model_vq, vla_pipe, tats_args, data_args, device)
    time_b = time.time()
    return instance_data




if __name__ == '__main__':
    parser = H4ArgumentParser((VLAModelArguments, DataArguments, TATSModelArguments))
    vla_args, data_args, tats_args = parser.parse()

    local_rank = os.getenv('LOCAL_RANK', 0)
    device = f'cuda:{local_rank}'

    assert tats_args.sequence_length == 6
    tats_args.weight_path = "/datassd_1T/model_infer/step_checkpoint-step_50000.ckpt"
    vla_args.model_name_or_path = "/datassd_1T/model_infer/mistral/checkpoint-40000/"

    model_vq = VQGANVisionActionEval(tats_args)
    state_dict = torch.load(tats_args.weight_path, map_location='cpu')['state_dict']
    result = model_vq.load_state_dict(state_dict, strict=False)
    for k in result.missing_keys:
        assert 'discriminator' in k or 'perceptual_model' in k
    model_vq = model_vq.eval().to(device)

    vla_pipe = mii.pipeline(vla_args.model_name_or_path)
    app.run(host='0.0.0.0',port=7777)