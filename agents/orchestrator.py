import os
from core.config import Config
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agents.tools.vision_tools import (
    query_past_detections, summarize_scene, set_search_intent, 
    calculate_route, save_object_signature, get_objects_near
)

class AgentEngine:
    """
    The Central Brain. This engine loads the 7 Agent Tool workflows and 
    maps Groq LLM natural language queries to execute the specific tool code.
    Uses pure LangChain Core Tool Binding (FAANG Standard).
    """
    def __init__(self, tts_callback=None, search_intent_callback=None):
        self.tts = tts_callback
        self.trigger_search = search_intent_callback
        self.llm_with_tools = None
        self.tools_map = {}

        if not Config.GROQ_API_KEY:
            print("  [Agent] LangChain/Groq disabled — Missing GROQ_API_KEY in .env")
            return

        print("  [Agent] Initializing Pure LangChain Core Reasoning Router...")
        try:
            self.llm = ChatGroq(
                temperature=0, 
                api_key=Config.GROQ_API_KEY, 
                model_name="llama-3.3-70b-versatile"
            )
            
            self.tools = [
                query_past_detections,
                summarize_scene,
                set_search_intent,
                calculate_route,
                save_object_signature,
                get_objects_near
            ]
            self.tools_map = {t.name: t for t in self.tools}
            self.llm_with_tools = self.llm.bind_tools(self.tools)
            
        except Exception as e:
            print(f"  [Agent] Integration Error: {e}")

    def process_voice_command(self, question: str, raw_scene_context: str):
        """
        Takes the user's microphone string, hands it the latest YOLO JSON background context,
        and tells the agent to think, act, and speak the result.
        """
        if not self.llm_with_tools:
            if self.tts:
                self.tts("My reasoning systems are currently offline. Please add an API key.")
            return

        prompt_str = (
            f"The user's current spatial camera raw data is: {raw_scene_context}\n\n"
            f"The user just asked you: '{question}'\n"
            "Use your tools if you need historical memory, routes, or target-setting. Provide the user with a single, comforting, descriptive sentence as your final answer."
        )

        messages = [
            SystemMessage(content="You are Visiona AI, the intelligent spatial guide for a completely blind user."),
            HumanMessage(content=prompt_str)
        ]

        try:
            # 1. Base Invoke (Agent thinks and possibly triggers tools)
            ai_msg = self.llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            # 2. Tool Execution Loop
            if ai_msg.tool_calls:
                for tool_call in ai_msg.tool_calls:
                    selected_tool = self.tools_map.get(tool_call["name"])
                    if selected_tool:
                        print(f"  [Agent] Triggered Workflow Tool: {tool_call['name']}")
                        tool_output = selected_tool.invoke(tool_call["args"])
                        messages.append(ToolMessage(tool_call_id=tool_call["id"], content=str(tool_output)))
                
                # 3. Final Answer Synthesis
                final_response = self.llm_with_tools.invoke(messages)
                text = final_response.content
            else:
                text = ai_msg.content

            print(f"  [Agent] Responded: {text}")
            if self.tts:
                self.tts(text)

        except Exception as e:
            print(f"  [Agent] Inference Error: {e}")
            if self.tts:
                self.tts("I am currently experiencing an error in my reasoning brain.")
