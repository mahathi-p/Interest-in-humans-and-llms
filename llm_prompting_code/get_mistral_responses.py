from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

device = "cuda"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
llm_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# llm_id = "mistralai/Mistral-7B-Instruct-v0.2"
# llm_id = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(llm_id,
                                             torch_dtype=torch.float16,
                                             device_map='auto'
                                            )


tokenizer = AutoTokenizer.from_pretrained(llm_id)

print("model loaded!")

tea_rates_begin = ["Can I please ask you to rate a student teacher conversation based on how interesting you feel the conversation is? Could you please provide an integer rating between 0(boring) to 4(interesting). It is a conversation between a student (2nd language English speaking) learning English from a teacher. Assume the role of the teacher and rate the student. Please give the Rating as an integer value in this format: INT: 'how interesting you find the reply of the student'; REASON: justify the rating ; EXP_INT: 'how interesting you would expect the next conversation to be';  REASON: justify the rating. This is the beginning the Conversation-"]
tea_rates_conv = ["Can I please ask you to rate a student teacher conversation based on how interesting you feel the conversation is? Could you please provide an integer rating between 0(boring) to 4(interesting). It is a conversation between a student (2nd language English speaking) learning English from a teacher. Assume the role of the teacher and rate the student. Please give the Rating as an integer value in this format: INT: 'how interesting you find the reply of the student'; REASON: justify the rating ; EXP_INT: 'how interesting you would expect the next conversation to be';  REASON: justify the rating. Keep the previous dialogue conversation in mind. The next dialogue is-"]

stu_rates_begin = ["Can I please ask you to rate a student teacher conversation based on how interesting you feel the conversation is? Could you please provide an integer rating between 0(boring) to 4(interesting). It is a conversation between a student (2nd language English speaking) learning English from a teacher. Assume the role of the student and rate the teacher. Please give the Rating as an integer value in this format: INT: 'how interesting you find the reply of the teacher'; REASON: justify the rating ; EXP_INT: 'how interesting you would expect the next conversation to be';  REASON: justify the rating. This is the beginning the Conversation-"]
stu_rates_conv = ["Can I please ask you to rate a student teacher conversation based on how interesting you feel the conversation is? Could you please provide an integer rating between 0(boring) to 4(interesting). It is a conversation between a student (2nd language English speaking) learning English from a teacher. Assume the role of the student and rate the teacher. Please give the Rating as an integer value in this format: INT: 'how interesting you find the reply of the teacher'; REASON: justify the rating ; EXP_INT: 'how interesting you would expect the next conversation to be';  REASON: justify the rating. Keep the previous dialogue conversation in mind. The next dialogue is-"]


conv_data = pd.read_excel("/home/m0hath1/TSCC_Data/FINAL_new_prompts.xlsx")
conv_data['first_role'] = conv_data['proc_text'].apply(lambda x : "Teacher" if x.startswith('teacher') else 'Student')
conv_data['second_role'] = conv_data['proc_text'].apply(lambda x : "Student" if x.startswith('teacher') else 'Teacher')


convs_prompts = []
prev_conv_id = 0

for idx, con in conv_data.iterrows():
    
    conv = con['proc_text']
    conv_id = con['conversation_id']
    first_role = con['first_role']
    second_role = con['second_role']

    if (first_role=='Teacher') & (conv_id!=prev_conv_id):
        prompt = tea_rates_begin[0] + conv
    elif (first_role=='Teacher') & (conv_id==prev_conv_id):
        prompt = tea_rates_conv[0] + conv
    elif (first_role=='Student') & (conv_id==prev_conv_id):
        prompt = stu_rates_begin[0] + conv
    else:
        prompt = stu_rates_conv[0] + conv

    prev_conv_id = con['conversation_id']

    convs_prompts.append(prompt)

print('prompts made')

first_reply = "INT: 1; REASON: The student's reply is a standard response during a conversation. It's not particularly interesting because it's expected in this context and doesn't reveal much about the student or their English language learning experience. However, as this is an initial greeting in the conversation, the interest level could possibly increase as the conversation progresses and specifics of the lesson are discussed. ; EXP_INT: 3; REASON: I foresee that the teacher will start engaging with some relevant topic or English language learning activities, which should make the conversation more interesting. Moreover, the language level of the student seemed good in the conversation, which raises my expectation for the next interaction."

messages = [
    {"role": "user", "content": "Can I please ask you to rate a student teacher conversation based on how interesting you feel the conversation is? Could you please provide an integer rating between 0(boring) to 4(interesting). It is a conversation between a student (2nd language English speaking) learning English from a teacher. Assume the role of the teacher and rate the student. Please give the Rating as an integer value(0,1,2,3,4) in this format: INT: 'how interesting you find the reply of the student'; REASON: justify the rating ; EXP_INT: 'how interesting you would expect the next conversation to be';  REASON: justify the rating. The conversation is: " + conv_data['proc_text'][0]},
    {"role": "assistant", "content": first_reply},
    {"role": "user", "content": "Can I please ask you to rate a student teacher conversation based on how interesting you feel the conversation is? Could you please provide an integer rating between 0(boring)) to 4(interesting). It is a conversation between a student (2nd language English speaking) learning English from a teacher. Assume the role of the teacher and rate the student. Please give the Rating as an integer value(0,1,2,3,4) in this format: INT: 'how interesting you find the reply of the student'; REASON: justify the rating ; EXP_INT: 'how interesting you would expect the next conversation to be';  REASON: justify the rating.  Please keep the previous conversation in mind. Next Dialogue is :" + conv_data['proc_text'][1]}

]
encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True,)
generated_ids = model.generate(encodeds, max_new_tokens=512, do_sample=True, pad_token_id=tokenizer.eos_token_id)
decoded = tokenizer.batch_decode(generated_ids)
    

ratings= []

ratings.append(first_reply)
    
print('first conv done, starting loop...')
print('first reply: ', decoded[0].split('[/INST]')[-1])

for i in range(2, len(convs_prompts)):
    assistant_reply = decoded[0].split('[/INST]')[-1]

    ratings.append(assistant_reply)
    messages.append(
        {
            "role": "assistant",
            "content": assistant_reply
        }
    )
    messages.append(
        {
            "role": "user",
            "content": convs_prompts[i]
        }
    )
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True,)
    generated_ids = model.generate(encodeds, max_new_tokens=512, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids)
    del messages[0:2]

    print(i)
    if i%500==0:
        print(i-1, " : ", assistant_reply)
        fileName = 'mix7x8_'+str(i)+'.pkl'
        with open(fileName, 'wb') as f:
            pickle.dump(ratings, f)
    
# adding the last reply
ratings.append(assistant_reply)

with open('mixtral_all1.pkl', 'wb') as f:
    pickle.dump(ratings, f)