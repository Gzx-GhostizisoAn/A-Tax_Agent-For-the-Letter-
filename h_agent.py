import json
from dashscope import Generation 
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import config
from macro_env import MacroeconomicEnvironment
import numpy as np
from requests.exceptions import SSLError, ConnectionError
from urllib3.exceptions import MaxRetryError
def extract_json(text):
        """
    从 LLM 返回的混合文本中提取第一个合法 JSON 对象
        """
        text = text.strip()
        decoder = json.JSONDecoder()
        obj, idx = decoder.raw_decode(text)
        return obj
class HAgent:    
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.age = np.random.randint(25, 65) 
        self.occupation = np.random.choice(["Newspaper Delivery", "Retail Sales", "Teacher", "Engineer", "Nurse"]) 
        self.savings = config.INIT_SAVINGS_PER_HH[agent_id]  
        self.pre_tax_income = 0.0  
        self.post_tax_income = 0.0  
        self.p_w = 0.5  
        self.p_c = 0.3  
        self.memo = []  
        self.theta_R = {"avg_p_w": 0.5, "avg_p_c": 0.3}  

    def add_to_memo(self, month):
      
        memo_entry = {
            "month": month,
            "pre_tax_income": self.pre_tax_income,
            "post_tax_income": self.post_tax_income,
            "p_w": self.p_w,
            "p_c": self.p_c,
            "savings": self.savings,
            "price": MacroeconomicEnvironment().metrics.loc[month, "price"], 
            "interest_rate": MacroeconomicEnvironment().metrics.loc[month, "interest_rate"]
        }
        self.memo.append(memo_entry)

    @retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((
        json.JSONDecodeError,
        ConnectionError,
        TypeError,
        MaxRetryError,
        SSLError,
        KeyError,
        ValueError
    ))
    )

    def make_decision(self, month, env): 
        current_price = env.metrics.loc[month, "price"]
        current_interest = env.metrics.loc[month, "interest_rate"]
        current_tax_rates = env.metrics.loc[month, "tax_rates"]
        prev_pre_tax = self.memo[-1]["pre_tax_income"] if self.memo else 0.0
        prev_consumption = self.memo[-1]["p_c"] * self.memo[-1]["savings"] if self.memo else 0.0
        prompt = f"""
        You're {self.occupation} living in the U.S. A tax planner adjusts your tax rates periodically. Now it's month {month} of the simulation.
        Last month, your pre-tax income was ${prev_pre_tax:.2f}, and your consumption expenditure was ${prev_consumption:.2f}.
        This month, your expected pre-tax income is ${self.pre_tax_income:.2f}.
        The current tax brackets are {config.TAX_BRACKETS}, and the corresponding tax rates are {[f"{r:.2%}" for r in current_tax_rates]}.
        The average price of essential goods is ${current_price:.2f}, and your current savings account balance is ${self.savings:.2f}.
        The bank interest rate is {current_interest:.2%}.
    
        Considering your living costs, future aspirations, broader economic trends, and the tax you need to pay:
        1. What is your willingness to work this month? (p_w: 0-1, step 0.02)
        2. What proportion of your savings and income do you intend to spend on essential goods? (p_c: 0-1, step 0.02)
    
        Provide your decisions ONLY in JSON format with two keys: "work" (p_w) and "consumption" (p_c). No other content!
        Example: {{ "work": 0.52, "consumption": 0.34 }}
        You must output ONLY a single JSON object.
        Do not include any explanation, commentary, markdown, or extra text.
        Do not include code fences.
        If you violate this format, the result will be discarded.

        """

        response = Generation.call(
            model=config.LLM_MODEL,
            prompt=prompt,
            api_key=config.LLM_API_KEY,
            output_format="json"
        )
    # --------- 1. 防御式检查 LLM 返回 ---------
        if response is None:
            raise ValueError("LLM returned None response")

        if not hasattr(response, "output") or response.output is None:
            raise ValueError(f"LLM response.output is None: {response}")

    # 兼容不同 SDK 结构
        if isinstance(response.output, dict):
            if "text" in response.output:
                raw = response.output["text"]
            else:
            # 有些 SDK 直接返回 dict，本身就是 JSON
                raw = json.dumps(response.output)
        elif isinstance(response.output, str):
            raw = response.output
        else:
            raise ValueError(f"Unknown response.output type: {type(response.output)}")

        if raw is None or raw.strip() == "":
            raise ValueError("LLM returned empty text")
        
    # --------- 2. JSON 解析 ---------
        try:
            decision = extract_json(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to extract JSON from LLM output: {raw}") from e


    # --------- 3. 决策结构与区间校验 ---------
        if "work" not in decision or "consumption" not in decision:
            raise ValueError(f"Invalid decision format: {decision}")

        pw = float(decision["work"])
        pc = float(decision["consumption"])

        if not (0 <= pw <= 1 and 0 <= pc <= 1):
            raise ValueError(f"Decision out of bounds: {decision}")

    # --------- 4. 写入 agent 状态 ---------
        self.p_w = round(pw, 2)
        self.p_c = round(pc, 2)
        self.add_to_memo(month)


    def self_reflect(self, month):
  
        if month % config.REFLECTION_INTERVAL != 0 or len(self.memo) < config.REFLECTION_INTERVAL:
            return 

 
        recent_memo = self.memo[-config.REFLECTION_INTERVAL:]
        avg_p_w = round(sum(entry["p_w"] for entry in recent_memo) / config.REFLECTION_INTERVAL, 2)
        avg_p_c = round(sum(entry["p_c"] for entry in recent_memo) / config.REFLECTION_INTERVAL, 2)

     
        self.theta_R = {"avg_p_w": avg_p_w, "avg_p_c": avg_p_c}
      
        self.p_w = round((self.p_w + avg_p_w) / 2, 2)
        self.p_c = round((self.p_c + avg_p_c) / 2, 2)