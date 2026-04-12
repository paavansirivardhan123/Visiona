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

    def extract_memory_label(self, text: str) -> dict:
        """
        Takes raw user speech like 'remember this, this is my wallet'
        and extracts the pure label ('my wallet') and the COCO base_class ('handbag').
        """
        if not self.llm:
            return None
            
        prompt = (
            f"The user is pointing their camera at something and said: '{text}'\n"
            "You must map their speech to two things in JSON format:\n"
            "1. 'alias': The exact human-readable name of the object or person. Fix grammar/spelling. If input is 'remember this he is ramesh', alias is 'Ramesh'. If input is 'it is my wallet', alias is 'my wallet'.\n"
            "2. 'base_class': The closest matching YOLO COCO class. The valid classes are: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.\n"
            "If it's 'Ramesh' or 'Sweater', base_class is 'person'. If 'my wallet', base_class is 'handbag' or 'backpack'.\n"
            "Output ONLY valid JSON like: {\"alias\": \"my wallet\", \"base_class\": \"handbag\"}\n"
            "Do not include markdown ticks."
        )
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            
            # Clean possible markdown block formatting
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            import json
            data = json.loads(content.strip())
            return data
        except Exception as e:
            print(f"  [Agent] Extract Model Error: {e}")
            return None

    def process_voice_command(self, question: str, raw_scene_context: str):
        """
        Takes the user's voice command, current scene context, and historical memory,
        and provides a natural language response.
        """
        if not self.llm_with_tools:
            if self.tts:
                self.tts("My reasoning systems are currently offline. Please add an API key.")
            return

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
            "1. DYNAMIC INTENT: If the user needs anything (rest, water, coffee, exit), use 'set_persistent_goal'.\n"
            "2. SHORTEST PATH NAVIGATION: If the user asks for directions or to find a place, use 'calculate_route'. The system will automatically find the SHORTEST walking path using their live location.\n"
            "3. BE PROACTIVE & VERBAL: When you provide directions, speak them EXACTLY as they are given to you. If you see a busy road (many vehicles), warn the user immediately.\n"
            "4. TOOL USAGE: Use tools for everything. If a goal is active and you see a candidate, guide the user to it.\n"
            "5. NATURAL TONE: Speak like a helpful friend. Provide a single, comforting, and EXTREMELY PRECISE sentence as your final response."
        )

        messages = [
            SystemMessage(content=(
                "You are a helpful spatial guide. You analyze raw vision data and memory to offer proactive assistance. "
                "If the user sounds tired, hungry, or lost, you scan your data for chairs, food, or exits and guide them. "
                "Your tone is comforting and your responses are always dynamic and spatially relevant."
            )),
            HumanMessage(content=prompt_str)
        ]

        try:
            # 1. Base Invoke (Agent thinks and possibly triggers tools)
            ai_msg = self.llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            # 2. Tool Execution Loop
            if getattr(ai_msg, 'tool_calls', None):
                for tool_call in ai_msg.tool_calls:
                    selected_tool = self.tools_map.get(tool_call["name"])
                    print(f"  [Agent] Tool call received: {tool_call}")
                    if selected_tool:
                        print(f"  [Agent] Triggered Workflow Tool: {tool_call['name']}")
                        # LangChain passes arguments as a dict, not a list
                        tool_args = tool_call.get("args", {})
                        if isinstance(tool_args, dict) and tool_args:
                            tool_output = selected_tool.invoke(tool_args)
                        else:
                            tool_output = selected_tool.invoke(tool_args)
                        messages.append(ToolMessage(tool_call_id=tool_call["id"], content=str(tool_output)))
                
                # 3. Final Answer Synthesis
                # Use the base LLM for synthesis to ensure we get a text response
                final_response = self.llm.invoke(messages)
                text = final_response.content
            else:
                text = ai_msg.content

            if text:
                print(f"  [Agent] Responded: {text}")
                if self.tts:
                    self.tts(text)
            else:
                print("  [Agent] Warning: LLM returned an empty response.")

        except Exception as e:
            print(f"  [Agent] Inference Error: {e}")
            if self.tts:
                self.tts("I am currently experiencing an error in my reasoning brain.")
