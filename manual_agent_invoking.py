from openai import OpenAI
from rag import search_documents, web_search, search_in_web_tool
from deepseek_ocr import extract_image_text
import json, re, os, copy
from dotenv import load_dotenv
load_dotenv()
URL = os.getenv("URL")


with open("instructions.txt") as t:
    instructions_text = t.read()

history_initial_template = [
    {
        "role": "system", 
        "content": f"{instructions_text}"
    }
]


def tool_return_nothing(history_log, tool_name, tool_request, additional_info = "absent."):
    history_log.extend([
                            {
                                "role": "assistant", 
                                "content": f"{tool_request}"
                            },
                            {
                                "role": "tool", 
                                "content": f'''
                                    [tool call] '{tool_name}' returned nothing.
                                            
                                    Additional information: {additional_info}
                                '''
                            }
                        ])

def tool_return_result(history_log, tool_name, tool_request, tool_result):
    history_log.extend([
                            {
                                "role": "assistant", 
                                "content": f"{tool_request}"
                            },
                            {
                                "role": "tool",
                                "content": f'''
                                    [tool call] '{tool_name}' returned following:
                                    
                                    {tool_result}
                                '''
                            }
                        ])


kazllm_history = copy.deepcopy(history_initial_template)
alemllm_history = copy.deepcopy(history_initial_template)

def extract_tool_call(text: str):
    match = re.search(r'\{.*"tool".*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def manual_agent_response(api_key, model, question, files_list = None):

    AVAILABLE_TOOLS = {"search_documents": search_documents, 
                        "web_search": web_search, 
                        "extract_image_text": extract_image_text,
                        "search_in_web_tool": search_in_web_tool}

    match model:
        case "alemllm":
            history_log = alemllm_history
        case "kazllm":
            history_log = kazllm_history
        case _:
            raise ValueError(f"Invalid model choice: {model}")
    
    if files_list is not None:
        history_log.append({"role": "system", "content": f"user uploaded following files: {files_list}" })

    history_log.append({"role": "user", "content": question})

    client = OpenAI(
        api_key=api_key,
        base_url=URL
    )

    MAX_ITERATIONS = 50
    for _ in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=model,
            messages=history_log,
            temperature=0.2
        )
        content = response.choices[0].message.content
        call = extract_tool_call(content)

        if not call:
            history_log.append({"role": "assistant", "content": content})
            return content

        tool_name = call["tool"]
        args = call.get("args", {})

        if tool_name not in AVAILABLE_TOOLS:
            history_log.append({"role": "tool", "content": f"[tool call] [error] tool with the name '{tool_name}' does not exist."})
            continue
        elif not args:
            history_log.append({"role": "tool", "content": f"[tool call] [error] you tried to use '{tool_name}' tool with empty arguments."})
            continue
        
        try:
            result = AVAILABLE_TOOLS[tool_name].invoke(args)
        except ValueError as e:
            history_log.append({"role": "system", "content": f"{e}"})
            continue

        if not result:
            history_log.append({"role": "tool", "content": f"[tool call] [error] '{tool_name}' tool returned nothing."})
            continue
            
        match tool_name:
            case "search_documents":

                #filter results by relevance
                tool_result = []
                for i in result:
                    if i.get('relevance_score', 0) > 0.001:
                        tool_result.append(i)
                
                print(f"[DEBUG] '{tool_name} found {len(tool_result)} relevant chunk/s.")

                #check if any relevant information is present
                if not tool_result:
                    print(f"[DEBUG] '{tool_name}' returned nothing.")
                    tool_return_nothing(history_log, tool_name, content)
                else:
                    tool_return_result(history_log, tool_name, content, tool_result)


            case "web_search":

                found_relevant_information = []

                for i in result:
                    if i.get("relevance_score", 0) > 0.3:
                        found_relevant_information.append(i)

                print(f"[DEBUG] '{tool_name} found {len(found_relevant_information)} relevant chunk/s.")

                if not found_relevant_information:
                    print(f"[DEBUG] No relevant information found by web_search.")
                    tool_return_nothing(history_log, tool_name, content, '''
                                                                Inform the user that provided link returned no relevant 
                                                                information regarding the given question. Ask if they want
                                                                to extract information from the link anyway.
                                                            ''')
                else:
                    print(f"[DEBUG] Relevant information found by web_search: {found_relevant_information}.")
                    tool_return_result(history_log, tool_name, content, found_relevant_information)


            case "extract_image_text":

                print(f"[DEBUG] '{tool_name}' was used.")
                tool_return_result(history_log, tool_name, content, result)


            case "search_in_web_tool":

                print(f"[DEBUG] '{tool_name}' was used.")
                tool_return_result(history_log, tool_name, content, result)


        if len(history_log) > 100:
            del history_log[2:10]
            history_log[1] = {
                "role": "system", 
                "content": "---------- History was cleared. ----------"
            }
            print(f"[DEBUG] '{history_log}' history was cleared.")