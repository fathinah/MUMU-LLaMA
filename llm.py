from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re
import json
model_name_or_path = "./Mixtral-8x7B-Instruct-v0.1-GPTQ/"
# To use a different branch, change revision
# For example: revision="gptq-4bit-128g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

def remove_extra_spaces(string):
    # Replace all occurrences of more than one space with a single space
    return re.sub(r'\s{2,}', ' ', string)
    
audio_caps = open('/l/users/fathinah.izzati/ml711/MU-LLaMA/MU-LLaMA/mullama_output1.json')
audio_caps = json.load(audio_caps)

video_caps = open('/l/users/xinyue.li/caption/SwinBERT/output/results.json')
video_caps = json.load(video_caps)

for audio in list(audio_caps.keys())[535:]:
    print(audio)
    prompt = f'''Video caption: '{video_caps[audio]['pred']}' \nMusic caption: '{audio_caps[audio]['pred']}' \nDescribe the music from both video and music captions. Return answer only without intro/outro.'''
    prompt_template=f'''[INST] {prompt} [/INST]
    '''
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    output = tokenizer.decode(output[0])
    output = output.split('[/INST]\n')[-1].replace('\n','').replace('</s>','').strip()
    output = remove_extra_spaces(output)
    outputs = {}
    outputs[audio]={'caption':output}
    x =  open('mixtral_output.json')
    z = json.load(x)
    z.update(outputs)
    with open('mixtral_output.json', 'w', encoding ='utf8') as x: 
        json.dump(z, x)