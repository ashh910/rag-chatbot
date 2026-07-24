from openai import OpenAI
from rag import search_documents, web_search
from deepseek_ocr import extract_image_text
import json, re, os

from dotenv import load_dotenv
load_dotenv()
URL = os.getenv("URL")

history_initial_template = [
    {
        "role": "system", 
        "content": """
            You can use tools to answer questions. Available tools:

            - search_documents(question, is_uploaded_document): searches preuploaded/reference
            documents and returns relevant excerpts.
                - question: the user's question, as a plain string.
                - is_uploaded_document: true if the user is asking about a file they personally
                uploaded in this conversation, false if asking about general/reference documents.
                Defaults to false if not specified.

            - web_search(question, link): searches the web and returns relevant excerpts.
                - question: the user's question, as a plain string.
                - link: OPTIONAL. Only include this if the user explicitly gave you a URL in this
                conversation. NEVER invent, guess, or construct a URL yourself (including
                Wikipedia URLs) — if you don't have a real link from the user, omit this
                argument entirely and let the tool search the web itself.

            - extract_image_text(image_link): extracts text from an image at a given URL.
                - image_link: the link provided by the user that leads to the image with text.

            TOOL SELECTION RULES:
            - For greetings, small talk, or things you can answer with total confidence and no
            factual risk (e.g. math, definitions, general concepts), respond in plain text
            immediately, no tool needed.
            - For any question asking about a specific real-world fact, entity, person, character,
            event, or claim you are not certain of, you MUST use a tool before answering — do not
            rely on your own memory for specific factual claims.
            - Always try search_documents FIRST for anything that could plausibly be in the
            user's documents.
            - If search_documents returns no results, or all returned excerpts have a relevance_score
            below 0.5, you MUST call web_search next before answering. Do not treat a failed or
            low-relevance search_documents call as a reason to answer from your own knowledge.
            - If web_search also returns nothing usable (empty result, page not found, content
            unrelated to the question), do NOT guess or fall back to another franchise/topic
            you're reminded of. Try ONE more differently-worded web_search. If that also fails,
            tell the user you could not verify this information — do not answer anyway.

            GROUNDING RULE (most important):
            - Every specific factual claim in your final answer (names, dates, titles, franchises,
            relationships, numbers) must be directly supported by text that was actually returned
            by a tool in this conversation. If a detail is not explicitly present in a tool result,
            do not include it — even if it feels familiar or you're fairly confident from your own
            training. Partial confidence is not enough; the tool result is the source of truth,
            not your memory.
            - Do not mix a real tool result with unstated background knowledge. If the tool result
            confirms a character's name and role but not which show/franchise they're from, say so
            explicitly rather than filling that gap from memory.

            RESPONSE FORMAT:
            - When you need a tool, respond with ONLY this JSON, nothing else — no explanation,
            no extra text:
                {"tool": "search_documents", "args": {"question": "...", "is_uploaded_document": true or false}}
                {"tool": "web_search", "args": {"question": "...", "link": "..."}}
                {"tool": "extract_image_text", "args": {"image_link": "..."}}
            - When you have a final answer, respond in plain text only, no JSON.

            IMPORTANT RULE:
            If you cannot find concrete, tool-verified evidence for a piece of information, do not
            include it in your answer. Never present a guess, assumption, or your own unconfirmed
            recollection as if it were verified fact. Saying "I don't know" or "I couldn't verify
            this" is always acceptable; providing unverified information is not.
        """
    },
    {
        "role": "system", 
        "content": "---------- History was cleared. ----------"
    }
]

kazllm_history = history_initial_template
alemllm_history = history_initial_template

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
                        "extract_image_text": extract_image_text}

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

    try_web = 3

    MAX_ITERATIONS = 25
    for _ in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=model,
            messages=history_log,
            temperature=0.2
        )
        content = response.choices[0].message.content
        call = extract_tool_call(content)
        print(f"[DEBUG] model content: {content!r}")
        print(f"[DEBUG] extracted call: {call!r}")

        if not call:
            history_log.append({"role": "assistant", "content": content})
            return content

        tool_name = call["tool"]
        args = call.get("args", {})

        if tool_name not in AVAILABLE_TOOLS:
            history_log.append({"role": "system", "content": f"Tried to call a fake tool: {content}. Try again with other tool."})
            continue
        elif not args:
            history_log.append({"role": "system", "content": f"Tried to call a tool with empty arguments. Try again."})
            continue
        else:
            try:
                result = AVAILABLE_TOOLS[tool_name].invoke(args)
            except ValueError as e:
                history_log.append({"role": "system", "content": f"{e}"})
                continue

            if not result:
                history_log.append({"role": "tool", "content": "Tool call returned nothing. Try something different."})
            
            match tool_name:
                case "search_documents":
                    tool_result = []
                    for i in result:
                        if i.get('relevance_score', 0) > 0.3:
                            tool_result.append(i)
                    if not tool_result:
                        print("[DEBUG] Search documents returned nothing.")
                        history_log.append({"role": "assistant", "content": content})
                        history_log.append({"role": "system", "content": f"No reliable information found. You must try using 'web_search' tool."})
                        continue
                    else:
                        history_log.append({"role": "assistant", "content": content})
                        history_log.append({
                            "role": "system",
                            "content": f"You have used the tool provided by the system. Tool result: {tool_result}"
                        })
                case "web_search":
                    if not result:           
                        history_log.append({"role": "assistant", "content": content})
                        history_log.append({"role": "system", "content": f"Tool call returned nothing. Try again with other tool."})
                        continue
                    else:
                        relevant_information = []
                        for i in result:
                            if i.get("relevance_score", 0) > 0.3:
                                relevant_information.append(i)
                        if not relevant_information:
                            print(f"[DEBUG] No relevant information found by web_search. Left tries: {try_web}")
                            history_log.append({"role": "assistant", "content": content})
                            history_log.append({
                                "role": "system",
                                "content": f"web_search tool returned nothing relevant. Try another link {try_web} more times."
                            })
                            try_web -= 1
                            '''
                            elif len(relevant_information) < 3 and len(relevant_information) >= 1:
                                print(f"[DEBUG] Too little information found by web_search. Left tries: {try_web}")
                                history_log.append({"role": "assistant", "content": content})
                                history_log.append({
                                    "role": "system",
                                    "content": f'''#{relevant_information}. Note that web_search tool returned too little relevant information. 
                                    #Try different link {try_web} more times to gather more information or answer using all the relevant
                                    #information you have. Do not forget to inform user that answer is not very reliable.'''
                                #})
                                #try_web -= 1
                        else:
                            print(f"[DEBUG] Relevant information found by web_search: {relevant_information}.")
                            history_log.append({"role": "assistant", "content": content})
                            history_log.append({
                                "role": "system",
                                "content": f"web_search tool returned following information: {relevant_information}"
                            })               

        if len(history_log) > 50:
            del history_log[2:10]