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
    Central reasoning engine. Routes voice commands to LangChain tools
    and synthesizes spoken responses via Groq LLaMA 3.3 70B.
    No confirmation step — commands execute immediately.
    """

    def __init__(self, tts_callback=None, search_intent_callback=None):
        self.tts = tts_callback
        self.trigger_search = search_intent_callback
        self.llm_with_tools = None
        self.tools_map = {}

        if not Config.GROQ_API_KEY:
            print("  [Agent] Groq disabled — GROQ_API_KEY missing in .env")
            return

        print("  [Agent] Initializing LangChain reasoning engine...")
        try:
            self.llm = ChatGroq(
                temperature=Config.LLM_TEMPERATURE,
                api_key=Config.GROQ_API_KEY,
                model_name=Config.LLM_MODEL,
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
                lower_goal_candidate_priority,
            ]
            self.tools_map = {t.name: t for t in self.tools}
            register_search_intent_callback(self.trigger_search)
            self.llm_with_tools = self.llm.bind_tools(self.tools)
            print("  [Agent] Ready.")
        except Exception as e:
            print(f"  [Agent] Init error: {e}")

    def process_voice_command(self, question: str, raw_scene_context: str):
        """
        Process a voice command immediately — no confirmation step.
        Routes to tools if needed, then speaks the response.
        """
        if not self.llm_with_tools:
            if self.tts:
                self.tts("Reasoning engine offline. Please add a Groq API key.")
            return

        # Direct memory hit — skip LLM for simple "where is X" queries
        from core.memory import memory_bank, goal_system
        custom_mem = memory_bank.find_custom_object(question)
        if custom_mem and "where" in question.lower():
            print(f"  [Agent] Memory hit: {custom_mem}")
            if self.tts:
                self.tts(custom_mem)
            return

        history_context = memory_bank.get_recent_history()
        goal_context    = goal_system.get_goal_summary()

        prompt_str = (
            "You are Visiona AI, a spatial guide for a blind user. Be brief and direct.\n\n"
            f"SCENE: {raw_scene_context}\n"
            f"MEMORY: {history_context}\n"
            f"GOALS: {goal_context}\n\n"
            f"USER SAID: '{question}'\n\n"
            "RULES:\n"
            "1. Act immediately — do not ask for confirmation.\n"
            "2. Use tools when needed (search, memory, GPS, goals).\n"
            "3. Keep your spoken response to one short sentence.\n"
            "4. If describing the scene, be compact: '2 persons ahead', 'Chair right'.\n"
            "5. If the user needs something (water, rest, exit), call set_persistent_goal."
        )

        messages = [
            SystemMessage(content=(
                "You are a helpful spatial guide for a blind user. "
                "Act on every request immediately without asking for confirmation. "
                "Use tools when appropriate. Respond in one short sentence."
            )),
            HumanMessage(content=prompt_str),
        ]

        try:
            ai_msg = self.llm_with_tools.invoke(messages)

            # Execute any tool calls immediately
            if getattr(ai_msg, "tool_calls", None):
                messages.append(ai_msg)
                for tool_call in ai_msg.tool_calls:
                    tool = self.tools_map.get(tool_call["name"])
                    if tool:
                        print(f"  [Agent] Tool: {tool_call['name']}")
                        output = tool.invoke(tool_call.get("args", {}))
                        messages.append(
                            ToolMessage(tool_call_id=tool_call["id"], content=str(output))
                        )

                # Synthesize final response from tool results
                final = self.llm_with_tools.invoke(messages)
                text  = final.content
            else:
                text = ai_msg.content

            if text:
                print(f"  [Agent] → {text}")
                if self.tts:
                    self.tts(text)

        except Exception as e:
            print(f"  [Agent] Error: {e}")
            if self.tts:
                self.tts("I encountered an error. Please try again.")
