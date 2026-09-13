"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import re
import sys
from typing import Any, Dict, List

from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""
    def query(self, user_input: str) -> Dict[str, Any]:
        """Return a one-shot answer and deliberately make no tool calls."""
        return {
            "status": "success",
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
        }

class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max(0, int(max_iterations))
        self.trace = []

    @staticmethod
    def _extract_codes(user_input: str) -> List[str]:
        """Extract supported airport/city codes in the order they occur."""
        return re.findall(r"\b(?:HAN|SGN|DAD)\b", user_input.upper())

    @staticmethod
    def _extract_budget(user_input: str) -> int:
        """Extract a VND ceiling, falling back to the tool's default budget."""
        text = user_input.lower().replace(",", ".")

        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:triệu|trieu|million|m)", text)
        if match:
            return int(float(match.group(1)) * 1_000_000)

        match = re.search(r"(\d+(?:\.\d+)?)\s*k\b", text)
        if match:
            return int(float(match.group(1)) * 1_000)

        match = re.search(r"(?:dưới|duoi|under|below)\s*(\d[\d\s]*)", text)
        if match:
            value = int(match.group(1).replace(" ", ""))
            # A plain number below 1000 is conventionally expressed in thousands.
            return value * 1_000 if value < 1_000 else value

        return 5_000_000

    @staticmethod
    def _has_any(text: str, *terms: str) -> bool:
        lowered = text.lower()
        return any(term in lowered for term in terms)

    def _plan(self, user_input: str) -> List[Dict[str, Any]]:
        """Create deterministic actions for the lab's offline data sources."""
        codes = self._extract_codes(user_input)
        is_flight_query = self._has_any(
            user_input, "chuyến bay", "chuyen bay", "vé máy bay", "ve may bay", "flight"
        ) and len(codes) >= 2
        is_weather_query = self._has_any(
            user_input, "thời tiết", "thoi tiet", "weather", "forecast"
        )

        actions: List[Dict[str, Any]] = []
        if is_flight_query:
            actions.append({
                "name": "get_flight_info",
                "args": {
                    "origin": codes[0],
                    "destination": codes[1],
                    "max_price": self._extract_budget(user_input),
                },
            })
        if is_weather_query:
            # In a flight + weather request the destination is the natural city.
            city_code = codes[-1] if codes else None
            if city_code:
                actions.append({
                    "name": "get_weather_forecast",
                    "args": {"city_code": city_code},
                })
        return actions

    @staticmethod
    def _execute_action(action: Any) -> Any:
        """Parse and execute an Action, returning a useful Observation."""
        if isinstance(action, str):
            try:
                action = json.loads(action)
            except (TypeError, ValueError):
                return {"error": "Invalid JSON format"}

        if not isinstance(action, dict):
            return {"error": "Invalid action format"}

        name = str(action.get("name", "")).strip().lower()
        args = action.get("args", {})
        if name not in TOOL_MAP:
            return {"error": f"Unknown tool: {name or '<empty>'}"}
        if not isinstance(args, dict):
            return {"error": "Invalid action arguments"}

        try:
            return TOOL_MAP[name](**args)
        except TypeError as exc:
            return {"error": f"Invalid arguments: {exc}"}
        except Exception as exc:  # Keep a bad tool from breaking the agent loop.
            return {"error": f"Tool execution failed: {exc}"}

    @staticmethod
    def _format_observation(observation: Any) -> str:
        return json.dumps(observation, ensure_ascii=False)

    def _final_answer(self, user_input: str, observations: List[Any]) -> str:
        """Turn collected observations into a concise customer-facing answer."""
        if not observations:
            return (
                "Vinpearl: Tôi chưa có dữ liệu chính sách đổi trả trong bộ dữ liệu này. "
                "Vui lòng liên hệ bộ phận hỗ trợ để được xác nhận."
            )

        parts: List[str] = []
        for observation in observations:
            if isinstance(observation, list):
                if not observation:
                    parts.append("Không tìm thấy chuyến bay phù hợp.")
                    continue
                flights = "; ".join(
                    f"{flight.get('flight_number', 'N/A')} ({flight.get('airline', 'N/A')}, "
                    f"{flight.get('departure_time', 'N/A')}, {flight.get('price_vnd', 'N/A')} VND)"
                    for flight in observation
                )
                parts.append(f"Các chuyến bay phù hợp: {flights}.")
            elif isinstance(observation, dict) and "temperature_c" in observation:
                parts.append(
                    f"Thời tiết {observation.get('city', 'không xác định')}: "
                    f"{observation['temperature_c']}°C, {observation.get('condition', '')}. "
                    f"{observation.get('recommendation', '')}"
                )
            elif isinstance(observation, dict) and "error" in observation:
                parts.append(f"Không thể lấy dữ liệu: {observation['error']}.")
            else:
                parts.append(str(observation))
        return " ".join(parts)

    def run(self, user_input: str) -> Dict[str, Any]:
        """Run the offline ReAct loop over the available tools."""
        self.trace = []
        actions = self._plan(user_input)
        observations: List[Any] = []
        iteration = 0

        if self.max_iterations == 0:
            return {
                "status": "max_iterations_reached",
                "answer": "Không thể hoàn thành trong số bước tối đa.",
                "iterations": 0,
                "trace": self.trace,
            }

        # One action is processed per iteration so every tool call is visible in the trace.
        while iteration < min(len(actions), self.max_iterations):
            iteration += 1
            action = actions[iteration - 1]
            observation = self._execute_action(action)
            observations.append(observation)
            self.trace.append({
                "step": iteration,
                "thought": "Thu thập dữ liệu cần thiết cho câu hỏi của khách hàng.",
                "action": action,
                "observation": observation,
            })

        if len(actions) > self.max_iterations:
            return {
                "status": "max_iterations_reached",
                "answer": "Không thể hoàn thành trong số bước tối đa.",
                "iterations": iteration,
                "trace": self.trace,
            }

        # A final-answer thought is folded into the last trace entry. This keeps the
        # requested one-step queries at one iteration while multi-step requests need
        # one final synthesis iteration.
        if len(actions) > 1:
            if iteration >= self.max_iterations:
                return {
                    "status": "max_iterations_reached",
                    "answer": "Không thể hoàn thành trong số bước tối đa.",
                    "iterations": iteration,
                    "trace": self.trace,
                }
            iteration += 1
            answer = self._final_answer(user_input, observations)
            self.trace.append({
                "step": iteration,
                "thought": "Đã đủ dữ liệu, tổng hợp câu trả lời cuối cùng.",
                "action": None,
                "observation": answer,
            })
        else:
            answer = self._final_answer(user_input, observations)
            if self.trace:
                self.trace[-1]["final_answer"] = answer
            else:
                self.trace.append({
                    "step": 1,
                    "thought": "Câu hỏi không cần tra cứu công cụ.",
                    "action": None,
                    "observation": answer,
                    "final_answer": answer,
                })
            iteration = 1

        return {
            "status": "completed",
            "answer": answer,
            "iterations": iteration,
            "trace": self.trace,
        }

def main():
    # Windows consoles may default to cp1252 while the lab data is UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"
    
    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
