import os
from core.config import Config
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agents.tools.vision_tools import (
    query_past_detections, summarize_scene, set_search_intent, 
    calculate_route, save_object_signature, get_objects_near,
    register_search_intent_callback, set_persistent_goal,
    mark_goal_completed, lower_goal_candidate_priority
)

class AgentEngine:
    """
    The Central Brain. This engine loads the 10 Agent Tool workflows and 
    maps Groq LLM natural language queries to execute the specific tool code.
    Uses pure LangChain Core Tool Binding (FAANG Standard).
    """
    def __init__(self, tts_callback=None, search_intent_callback=None):
        self.tts = tts_callback
        self.trigger_search = search_intent_callback
        self.llm_with_tools = None
        self.tools_map = {}
        
        # Confirmation System
        self._pending_tool_calls = []
        self._pending_message = ""
        self._is_awaiting_confirmation = False

        if not Config.GROQ_API_KEY:
            print("  [Agent] LangChain/Groq disabled — Missing GROQ_API_KEY in .env")
            return

        print("  [Agent] Initializing Pure LangChain Core Reasoning Router...")
        try:
            model_name = getattr(Config, "LLM_MODEL", "llama-3.3-70b-versatile")
            self.llm = ChatGroq(
                temperature=Config.LLM_TEMPERATURE, 
                api_key=Config.GROQ_API_KEY, 
                model_name=model_name
            )
            
            self.tools = [
                query_past_detections,
                summarize_scene,
                set_search_intent,
                calculate_route,
                save_object_signature,
                get_objects_near,
                set_persistent_goal,
                mark_goal_completed,
                lower_goal_candidate_priority
            ]
            self.tools_map = {t.name: t for t in self.tools}
            register_search_intent_callback(self.trigger_search)
            self.llm_with_tools = self.llm.bind_tools(self.tools)
            
        except Exception as e:
            print(f"  [Agent] Integration Error: {e}")

    def process_voice_command(self, question: str, raw_scene_context: str):
        """
        Takes the user's voice command, current scene context, and historical memory,
        and provides a natural language response.
        """
        if not self.llm_with_tools:
            if self.tts:
                self.tts("My reasoning systems are currently offline. Please add an API key.")
            return

        # 1. Handle Confirmation Flow
        if self._is_awaiting_confirmation:
            if any(word in question.lower() for word in ["yes", "yeah", "sure", "ok", "correct"]):
                print("  [Agent] User confirmed intent. Executing pending tools...")
                self._execute_pending_tools()
                return
            elif any(word in question.lower() for word in ["no", "stop", "cancel", "wrong"]):
                print("  [Agent] User cancelled intent.")
                self._is_awaiting_confirmation = False
                self._pending_tool_calls = []
                if self.tts: self.tts("Okay, I've cancelled that request. How else can I help?")
                return
            # If it's not a clear yes/no, we fall through to normal processing

        # Fetch historical memory context
        from core.memory import memory_bank, goal_system
        history_context = memory_bank.get_recent_history()
        goal_context = goal_system.get_goal_summary()

        # Quick check for custom objects in memory before calling LLM
        custom_mem = memory_bank.find_custom_object(question)
        if custom_mem and "where" in question.lower():
             print(f"  [Agent] Direct memory hit: {custom_mem}")
             if self.tts: self.tts(custom_mem)
             return

        prompt_str = (
            "You are Visiona AI, a natural spatial guide for the blind. You reason like a human.\n\n"
            f"SPATIAL DATA: {raw_scene_context}\n"
            f"MEMORY: {history_context}\n"
            f"PERSISTENT GOALS: {goal_context}\n\n"
            f"USER SAID: '{question}'\n\n"
            "INSTRUCTIONS:\n"
            "1. VERIFY INTENT: Always start your response by repeating the user's core request in a short question. "
            "Example: 'Asking for water?' or 'Asking for coffee?'. Use this format exactly.\n"
            "2. COMPACT SCENE: If describing the environment, keep it extremely brief. "
            "Example: '2 persons ahead' or 'Chair right'. Do not use distances or steps unless specifically asked.\n"
            "3. SPELLING & CLARITY: If the user's request has obvious spelling errors, suggest the correct word in your verification question.\n"
            "4. DYNAMIC INTENT: If the user needs anything (rest, water, exit), use 'set_persistent_goal'.\n"
            "5. NATURAL TONE: Speak like a helpful friend. Provide a single, short, and easy to understand sentence."
        )

        messages = [
            SystemMessage(content=(
                "You are a helpful spatial guide. You analyze raw vision data and memory to offer proactive assistance. "
                "If the user sounds tired, hungry, or lost, you scan your data for chairs, food, or exits and guide them. "
                "If the user's request is unclear or has typos, you must ask for confirmation before acting."
            )),
            HumanMessage(content=prompt_str)
        ]

        try:
            # 1. Base Invoke (Agent thinks and possibly triggers tools)
            ai_msg = self.llm_with_tools.invoke(messages)
            
            # Check for tool calls
            if getattr(ai_msg, 'tool_calls', None):
                # If the AI wants to use tools, we check if it ALSO provided a confirmation question in the text
                # Or we decide to force confirmation for certain important tools
                self._pending_tool_calls = ai_msg.tool_calls
                
                # We ask the LLM to generate a confirmation message based on these tool calls
                confirm_prompt = f"The user said '{question}'. I want to call these tools: {ai_msg.tool_calls}. " \
                                 f"Generate an EXTREMELY SHORT confirmation question in the format: 'Asking for [intent]?' " \
                                 f"If it's a simple, clear request, just say 'CLEAR'."
                
                check_msg = self.llm.invoke([SystemMessage(content="You generate 3-word confirmation questions."), HumanMessage(content=confirm_prompt)])
                
                if "CLEAR" not in check_msg.content.upper():
                    self._is_awaiting_confirmation = True
                    print(f"  [Agent] Awaiting confirmation for: {ai_msg.tool_calls}")
                    if self.tts: self.tts(check_msg.content)
                    return
                else:
                    # Clear intent, execute immediately
                    self._execute_pending_tools()
            else:
                # No tool calls, just respond
                if ai_msg.content:
                    print(f"  [Agent] Responded: {ai_msg.content}")
                    if self.tts: self.tts(ai_msg.content)

        except Exception as e:
            print(f"  [Agent] Inference Error: {e}")
            if self.tts: self.tts("I am currently experiencing an error in my reasoning brain.")

    def _execute_pending_tools(self):
        """Executes the tool calls stored in self._pending_tool_calls."""
        if not self._pending_tool_calls:
            return
            
        try:
            self._is_awaiting_confirmation = False
            results_for_llm = []
            
            for tool_call in self._pending_tool_calls:
                selected_tool = self.tools_map.get(tool_call["name"])
                if selected_tool:
                    print(f"  [Agent] Executing confirmed tool: {tool_call['name']}")
                    tool_args = tool_call.get("args", {})
                    tool_output = selected_tool.invoke(tool_args)
                    results_for_llm.append(ToolMessage(tool_call_id=tool_call["id"], content=str(tool_output)))
            
            # Synthesize final response after tool execution
            # We use the LLM to summarize the tool results into one short sentence
            final_prompt = f"I have executed these actions for the user: {results_for_llm}. " \
                           f"Provide a 1-sentence confirmation that the task is done or the search has started."
            
            final_resp = self.llm.invoke([SystemMessage(content="You are a concise assistant."), HumanMessage(content=final_prompt)])
            
            if self.tts: self.tts(final_resp.content)
            self._pending_tool_calls = []
            
        except Exception as e:
            print(f"  [Agent] Execution Error: {e}")
