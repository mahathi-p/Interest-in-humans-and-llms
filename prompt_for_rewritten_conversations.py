import openai
import tqdm
import json
import time
from datetime import datetime
from argparse import ArgumentParser
from os.path import isfile


SYSTEM_MESSAGES = {
        "student": "Imagine you are a student trying to learn English. You are having a conversation with the user who is an English teacher.",
        "teacher": "Imagine you are an English teacher. You are having a conversation with the user who is an English learner."
    }

SYSTEM_MESSAGES_bored = {
        "teacher": "You are a bad English teacher who always makes students bored.",
        "student": "You are a student who lacks enthusiasm and doesn't want to study anymore."
    }

DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant"

BORING_PROMT = "Instruction: given a text message from a teaching session between a teacher and a student, please provide a more straightforward and less engaging version. Strip away any colorful language or additional context to make the message as boring as possible. Please keep the main information from the message."
BORING_PROMPT_1 = "Please rewrite the following message into a more boring or mundane version:"
PROMPT_HEAD = "Below is an informal conversation between an English teacher and a student in an English lesson."
PROMPT_TAIL = "Instruction: Could you please rewrite the last message of the <role>? Please preserve the essence of the information conveyed, particularly the underscores from any \"fill in the blank\" exercises mentioned, but make the rewritten messages as boring and emotionless as possible. Do not rewrite any other messages."
PROMPT_HEAD_2 = "Below is an informal conversation between an English teacher and a student in an English lesson. Please rewrite the last message from the <role>. Your rewrite should preserve the essence of the information conveyed but it should aim to make the <role>'s message as dry and emotionless as possible. Please preserve the underscores from any \"fill in the blank\" exercises mentioned in the original message. Do not rewrite any other messages."

def get_cid(this_dict):
    if this_dict.get("conversation_id") is None:
        cid = this_dict["coversation_id"]
    else:
        cid = this_dict["conversation_id"]
    return cid


def verify_message(message, role):
    temp_get_role = message.split(":")
    if len(temp_get_role) > 2:
        temp_get_role = [temp_get_role[-2][-7:], temp_get_role[-1]]
    


if __name__ == "__main__":
    parser = ArgumentParser(description='Query openai for graphs')
    parser.add_argument("--path", type=str, default="merged/Project_10_mod.json",
            help="the path to the processed data file")
    args = parser.parse_args()

    openai.api_key = "key"

    with open(args.path, 'r') as f:
        old_con = json.loads(f.read())

    prompt_list = []
    conv_cache = []
    for i in range(len(old_con)):
        if old_con[i]["text"][:11] == "===========":
            prompt_list.append("None")
            conv_cache = []
        else:
            speaker1 = old_con[i]['text'].split("========")[0]
            speaker2 = old_con[i]['text'].split("========")[-1]

            conv_cache.append(speaker1)
            conv_cache.append(speaker2)
            start = max(0, len(conv_cache) - 9)
            message_history = ' \n'.join(conv_cache[start:])
            speaker_head = speaker2.split(':')[0].replace("\n", "")
            this_tail = PROMPT_TAIL.replace("<role>", speaker_head)

            #this_head = PROMPT_HEAD_2.replace("<role>", speaker_head)
            #content = ':'.join(speaker2.split(':')[1:])
            this_prompt = f"{PROMPT_HEAD} \n{message_history} \n\n{this_tail}"
            #this_prompt = f"{this_head} \n{message_history}"
            prompt_list.append(this_prompt)

    with open("temp_prompt_list.json", 'w') as f:
        f.write(json.dumps(prompt_list, indent=4))

    #################

    this_temperature = 0.3
    name_base = args.path.split("/")[0]
    name = args.path.split("/")[-1]
    name = name.split(".")[0]
    
    save_file_path = f"json/{name}.json"

    request_model = "gpt-3.5-turbo"

    gpt4_call_count = 0

    if not isfile(save_file_path):
        new_con = []
    else:
        with open(save_file_path, 'r') as f:
            new_con = json.loads(f.read())
    while len(new_con) < len(old_con):
        print(len(new_con))
        if old_con[len(new_con)]["text"][:11] == "===========":
            new_con.append(old_con[len(new_con)])
        else:
            speaker1 = old_con[len(new_con)]['text'].split("========")[0]
            speaker2 = old_con[len(new_con)]['text'].split("========")[-1]

            target_role = speaker2.split(":")[0].replace("\n", "")

            target_text = speaker2.split(":")[1].split()
            if len(target_text) < 2:
                new_text = "[No alternative, please choose a comparison label randomly]"
                new_con.append(
                    {"label": old_con[len(new_con)]["label"],
                    "conversation_id": get_cid(old_con[len(new_con)]),
                    "text": new_text}
                )
                continue

            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE}, #SYSTEM_MESSAGES_bored[speaker2.lower().split(":")[0].replace("\n", "")]},
                {"role": "user", "content": prompt_list[len(new_con)]},
            ]
            print(f"prompting {request_model}...")
            try:
                respond =  openai.ChatCompletion.create(model=request_model, messages=messages, temperature=this_temperature)
            except:
                error_message = "Error occurred! " + str(datetime.now())
                print(error_message)
                time.sleep(60)
                continue

            time.sleep(3)
            rewrite_messages = respond["choices"][0]["message"]["content"]

            ########verify
            rewrite_messages = rewrite_messages.replace("TEACHER:", "TEACHER*")
            rewrite_messages = rewrite_messages.replace("STUDENT:", "STUDENT*")
            rewrite_messages = rewrite_messages.replace("teacher:", "TEACHER*")
            rewrite_messages = rewrite_messages.replace("student:", "STUDENT*")
            rewrite_messages = rewrite_messages.replace(":", "")
            rewrite_messages = rewrite_messages.replace("TEACHER*", "TEACHER:")
            rewrite_messages = rewrite_messages.replace("STUDENT*", "STUDENT:")

            temp_get_role = rewrite_messages.split(":")
            if len(temp_get_role) > 2:
                temp_get_role = [temp_get_role[-2][-7:], temp_get_role[-1]]

            if len(temp_get_role) > 1:
                if temp_get_role[0].lower() != target_role.lower():
                    print(prompt_list[len(new_con)])
                    print(rewrite_messages)
                    print(f"Error! Role mismatch! {this_temperature}")
                    this_temperature += 0.1
                    if this_temperature > 0.6 and request_model == "gpt-3.5-turbo":
                        this_temperature = 0.3
                        request_model = "gpt-4-1106-preview"
                        print("using gpt-4")
                        gpt4_call_count += 1
                    continue
                else:
                    rewrite_messages = ':'.join(temp_get_role)
            else:
                rewrite_messages = f"{target_role}: {rewrite_messages}"

            rewrite_messages = rewrite_messages.replace("]", "")
            rewrite_messages = rewrite_messages.replace("[", "")

            if "\nInstruction" in rewrite_messages:
                rewrite_messages = rewrite_messages.split("\nInstruction")[0]
            
            if "rewritten" in rewrite_messages:
                rewrite_messages = rewrite_messages.split("rewritten")[1]
                rewrite_messages = f"{target_role}:\n    {rewrite_messages}"
            
            if "REWRITTEN" in rewrite_messages:
                rewrite_messages = rewrite_messages.split("REWRITTEN")[1]
                rewrite_messages = f"{target_role}:\n    {rewrite_messages}"

            new_text = f"{speaker1}\n========Rate if this student finds the teacher interesting (please don't use your own preferences)========\n{rewrite_messages}"
            new_con.append(
                {"label": old_con[len(new_con)]["label"],
                "conversation_id": get_cid(old_con[len(new_con)]),
                "text": new_text}
            )
            this_temperature = 0.3
            request_model = "gpt-3.5-turbo"

        with open(save_file_path, 'w') as f:
            f.write(json.dumps(new_con, indent=4))
    
    print(f"File was saved at {save_file_path}")
    print(f"Total gpt-4 calls: {gpt4_call_count}")

