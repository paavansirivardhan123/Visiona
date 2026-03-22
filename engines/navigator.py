import os
import queue
import threading
import time
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.config import Config
from models.detection import Detection
from engines.memory import SceneMemory

class AINavigator:
    """
    AI-powered navigation engine with:
    - 5-zone spatial awareness
    - Real distance-based safety rules
    - Approaching object detection
    - Scene memory + LLM context
    - Smart routing (left/right/back fallback)
    - Target-seeking mode
    """

    def __init__(self):
        load_dotenv()
        self.intent = "walk forward"
        self.current_state = "SCANNING"
        self.reasoning = "Initializing NaVision AI..."
        self.last_llm_time = 0.0
        self.last_detections: List[Detection] = []
        self.memory = SceneMemory()
        self._llm_queue: queue.Queue = queue.Queue()
        self._init_langchain()
        self._thread = threading.Thread(target=self._llm_worker, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ #
    #  LangChain Setup
    # ------------------------------------------------------------------ #

    def _init_langchain(self):
        if not os.getenv("GROQ_API_KEY"):
            print("Warning: GROQ_API_KEY not set. LLM reasoning disabled.")
            self.llm = None
            return
        try:
            self.llm = ChatGroq(model=Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE)

            # Navigation reasoning prompt
            self.prompt = ChatPromptTemplate.from_template("""
You are NaVision, an AI navigation co-pilot for a visually impaired user.

USER INTENT: {intent}
CURRENT SCENE: {scene}
APPROACHING OBJECTS: {approaching}
SCENE HISTORY: {history}
PREVIOUS REASONING: {last_reasoning}

STRICT RULES:
1. SAFETY FIRST — if a hazard is within 80cm ahead, say STOP immediately.
2. Be CONCISE — max 2 short sentences, plain spoken English.
3. Be SPECIFIC — mention object names and directions.
4. If user is searching for a target, guide them toward it.
5. If path is clear, encourage forward movement confidently.

Respond ONLY in this format:
REASONING: [your internal logic] | INSTRUCTION: [what to say aloud to the user]
""")
            self.chain = self.prompt | self.llm | StrOutputParser()

            # Free-form question prompt
            self.question_prompt = ChatPromptTemplate.from_template("""
You are NaVision, an AI assistant for a visually impaired user.

CURRENT SCENE: {scene}
USER QUESTION: {question}

Answer helpfully and concisely in 1-2 spoken sentences.
Only use what you can infer from the scene description.
""")
            self.question_chain = self.question_prompt | self.llm | StrOutputParser()

        except Exception as e:
            print(f"LLM init warning: {e}")
            self.llm = None

    # ------------------------------------------------------------------ #
    #  LLM Worker Thread
    # ------------------------------------------------------------------ #

    def _llm_worker(self):
        while True:
            data = self._llm_queue.get()
            if data is None:
                break
            scene_desc, approaching_desc, speech_cb = data
            self._run_llm(scene_desc, approaching_desc, speech_cb)
            self._llm_queue.task_done()

    def _run_llm(self, scene_desc: str, approaching_desc: str, speech_cb):
        if not self.llm:
            return
        try:
            response = self.chain.invoke({
                "intent": self.intent,
                "scene": scene_desc,
                "approaching": approaching_desc,
                "history": self.memory.get_context_summary(),
                "last_reasoning": self.reasoning,
            })
            if "|" in response:
                reason_part, instruct_part = response.split("|", 1)
                self.reasoning = reason_part.replace("REASONING:", "").strip()
                instruction = instruct_part.replace("INSTRUCTION:", "").strip()
                speech_cb(instruction)
            else:
                self.reasoning = response.strip()
        except Exception as e:
            self.reasoning = f"AI error: {e}"

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def set_intent(self, intent: str, speech_cb):
        self.intent = intent.lower()
        self.current_state = "SEARCHING"
        self.reasoning = f"Searching for {intent}..."
        speech_cb(f"New goal: {intent}", priority=True)

    def process_frame(self, detections: List[Detection], speech_cb):
        self.last_detections = detections

        # Update scene memory
        detections = self.memory.update(detections)

        # 1. Fast path — safety rules
        instruction, is_priority = self._check_safety(detections, speech_cb)
        if instruction:
            speech_cb(instruction, priority=is_priority)
            return

        # 2. Target seeking
        if self.intent not in ("walk forward",):
            found = self._seek_target(detections, speech_cb)
            if found:
                return

        # 3. Slow path — LLM reasoning
        now = time.time()
        if self.llm and (now - self.last_llm_time > Config.LLM_COOLDOWN):
            if self._llm_queue.empty():
                scene_desc = self._describe_scene(detections)
                approaching = self._describe_approaching()
                self.memory.add_scene_description(scene_desc)
                self._llm_queue.put((scene_desc, approaching, speech_cb))
                self.last_llm_time = now

    # ------------------------------------------------------------------ #
    #  Safety Rules (Fast Path)
    # ------------------------------------------------------------------ #

    def _check_safety(self, detections: List[Detection], speech_cb) -> Tuple[Optional[str], bool]:
        ahead = [d for d in detections if d.position == "ahead"]
        left  = [d for d in detections if d.position in ("left", "far-left")]
        right = [d for d in detections if d.position in ("right", "far-right")]

        # Critical hazard directly ahead
        critical = [d for d in ahead if d.distance_cm and d.distance_cm < Config.DIST_CRITICAL]
        if critical:
            obj = critical[0]
            self.current_state = "ALERT"
            self.reasoning = f"Critical: {obj.label} at {int(obj.distance_cm)}cm ahead."
            # Spatial beep handled by caller via speech engine
            return f"Stop! {obj.label} is {int(obj.distance_cm)} centimeters ahead.", True

        # Near hazard ahead — find best escape route
        near_ahead = [d for d in ahead if d.distance_cm and d.distance_cm < Config.DIST_NEAR]
        if near_ahead:
            self.current_state = "AVOIDING"
            obj = near_ahead[0]
            left_clear  = not any(d.distance_cm and d.distance_cm < Config.DIST_NEAR for d in left)
            right_clear = not any(d.distance_cm and d.distance_cm < Config.DIST_NEAR for d in right)

            if left_clear:
                self.reasoning = f"Avoiding {obj.label}, turning left."
                return f"{obj.label} ahead at {int(obj.distance_cm)} centimeters. Turn left.", False
            elif right_clear:
                self.reasoning = f"Avoiding {obj.label}, turning right."
                return f"{obj.label} ahead. Turn right.", False
            else:
                self.reasoning = "All paths blocked."
                return "Path blocked on all sides. Please stop and wait.", True

        # Approaching objects (moving toward user)
        approaching = self.memory.get_approaching()
        approaching_ahead = [d for d in approaching if d.position == "ahead"]
        if approaching_ahead:
            obj = approaching_ahead[0]
            self.current_state = "ALERT"
            self.reasoning = f"{obj.label} is approaching."
            return f"Caution. {obj.label} is moving toward you.", False

        self.current_state = "SCANNING"
        self.reasoning = "Path clear. Proceeding."
        return None, False

    # ------------------------------------------------------------------ #
    #  Target Seeking
    # ------------------------------------------------------------------ #

    def _seek_target(self, detections: List[Detection], speech_cb) -> bool:
        target_label = self.intent.lower()
        matches = [d for d in detections if target_label in d.label.lower()]
        if not matches:
            return False

        target = min(matches, key=lambda d: d.distance_cm or 9999)
        self.current_state = "GUIDING"
        dist_str = f"{int(target.distance_cm)} centimeters" if target.distance_cm else "nearby"
        direction = target.position.replace("-", " ")
        self.reasoning = f"Target {target.label} found {direction}, {dist_str}."
        speech_cb(f"Found {target.label}. It is {dist_str} to your {direction}.")
        return True

    # ------------------------------------------------------------------ #
    #  Scene Description
    # ------------------------------------------------------------------ #

    def _describe_scene(self, detections: List[Detection]) -> str:
        if not detections:
            return "Clear path, no objects detected."
        parts = []
        for d in detections[:6]:  # Limit to 6 most relevant
            dist = f"{int(d.distance_cm)}cm" if d.distance_cm else "unknown dist"
            parts.append(f"{d.label} ({d.position}, {dist})")
        return ", ".join(parts)

    def _describe_approaching(self) -> str:
        approaching = self.memory.get_approaching()
        if not approaching:
            return "None"
        return ", ".join(f"{d.label} from {d.position}" for d in approaching)

    def stop(self):
        self._llm_queue.put(None)

    def answer_question(self, question: str, speech_cb):
        """Handle a free-form spoken question from the user."""
        if not self.llm:
            speech_cb("AI is not available right now.")
            return
        scene = self._describe_scene(self.last_detections)
        # Run in LLM thread to avoid blocking
        def _ask():
            try:
                response = self.question_chain.invoke({
                    "scene": scene,
                    "question": question,
                })
                speech_cb(response.strip(), priority=False)
            except Exception as e:
                speech_cb("Sorry, I could not answer that.")
        threading.Thread(target=_ask, daemon=True).start()
