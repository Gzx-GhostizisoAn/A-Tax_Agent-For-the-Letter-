
import os
import random
from dotenv import load_dotenv

load_dotenv()

SIMULATION_MONTHS = 120
NUM_HOUSEHOLDS = 50    
TAX_BRACKETS = [0.00, 808.33, 3289.58, 7016.67, 13393.75, 17008.33, 42525.00] 
PRODUCTIVITY = 1.0       

INIT_PRICE = 126.78       
INIT_INVENTORY = 0.0     
INIT_INTEREST_RATE = 0.03 
INIT_SAVINGS_PER_HH = [round(random.uniform(10000, 30000), 2) for i in range(NUM_HOUSEHOLDS)]  
INIT_WAGE = 20.0         


LLM_ENABLED = True
LLM_PROVIDER = "dashscope"
LLM_MODEL = "qwen-turbo-2024-09-19"

LLM_API_KEY = "Your Key"
REFLECTION_INTERVAL = 3        

BASELINE_TAX_RATES = {
    "us_federal": [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37],  
    "saez": [0.15, 0.20, 0.25, 0.30, 0.38, 0.45, 0.50],       
    "free_market": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]  
}