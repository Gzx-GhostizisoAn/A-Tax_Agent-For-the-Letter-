import pandas as pd
import numpy as np
import random
from config import SIMULATION_MONTHS, INIT_PRICE, INIT_INVENTORY, INIT_INTEREST_RATE, INIT_WAGE, NUM_HOUSEHOLDS, TAX_BRACKETS, PRODUCTIVITY


class MacroeconomicEnvironment:
    def __init__(self):
   
        tax_default = [0.0] * len(TAX_BRACKETS)
        self.metrics = pd.DataFrame({
            "month": list(range(SIMULATION_MONTHS)),
            "price": [INIT_PRICE] * SIMULATION_MONTHS,
            "inventory": [INIT_INVENTORY] * SIMULATION_MONTHS,
            "interest_rate": [INIT_INTEREST_RATE] * SIMULATION_MONTHS,
            "inflation": [0.0] * SIMULATION_MONTHS,
            "unemployment": [0.0] * SIMULATION_MONTHS,
            "equality": [0.0] * SIMULATION_MONTHS,
            "productivity": [0.0] * SIMULATION_MONTHS,

            "tax_rates": [tax_default.copy() for _ in range(SIMULATION_MONTHS)]
        })
        self.current_wage = INIT_WAGE

    def calculate_total_labor_supply(self, h_agents):
        total_labor = 0.0
        for agent in h_agents:
            total_labor += agent.p_w * 168 * PRODUCTIVITY
        return total_labor

    def update_inventory_after_production(self, month, total_labor):
        prev_inventory = self.metrics.loc[month - 1, "inventory"] if month > 0 else INIT_INVENTORY
        self.metrics.loc[month, "inventory"] = prev_inventory + total_labor

    def calculate_pre_tax_income(self, h_agents, total_labor):
        total_output_value = total_labor * self.current_wage
        pre_tax_incomes = []
        for agent in h_agents:
            labor_share = (agent.p_w * 168 * PRODUCTIVITY) / total_labor if total_labor > 0 else 0.0
            pre_tax = labor_share * total_output_value
            pre_tax = round(pre_tax, 2)
            pre_tax_incomes.append(pre_tax)
            agent.pre_tax_income = pre_tax
        return pre_tax_incomes

    def calculate_tax_and_redistribution(self, month, h_agents):
        current_tax_rates = list(self.metrics.loc[month, "tax_rates"])
        total_tax = 0.0
        tax_records = []
        for agent in h_agents:
            pre_tax = agent.pre_tax_income
            tax = 0.0
            for i in range(len(TAX_BRACKETS) - 1):
                b_low = TAX_BRACKETS[i]
                b_high = TAX_BRACKETS[i + 1]
                tau = current_tax_rates[i] if i < len(current_tax_rates) else 0.0
                if pre_tax <= b_low:
                    break
                taxable = min(pre_tax, b_high) - b_low
                tax += taxable * tau
            tax = round(tax, 2)
            tax_records.append(tax)
            total_tax += tax

        redistribution_per_hh = round(total_tax / NUM_HOUSEHOLDS, 2) if NUM_HOUSEHOLDS > 0 else 0.0
        for i, agent in enumerate(h_agents):
            post_tax_income = agent.pre_tax_income - tax_records[i] + redistribution_per_hh
            agent.post_tax_income = round(post_tax_income, 2)

    def update_consumption_and_inventory(self, month, h_agents):
        current_price = self.metrics.loc[month, "price"]
        prev_inventory = self.metrics.loc[month, "inventory"]
        consumption_demands = []
        for agent in h_agents:
            demand = (agent.p_c * agent.savings) / (current_price if current_price > 0 else 1.0)
            consumption_demands.append(round(demand, 2))

        random_indices = np.random.permutation(len(h_agents))
        remaining_inventory = prev_inventory

        for idx in random_indices:
            agent = h_agents[int(idx)]
            demand = consumption_demands[int(idx)]
            actual_consumption = min(demand, remaining_inventory)
            actual_consumption_cost = round(actual_consumption * current_price, 2)
            interest = round(agent.savings * self.metrics.loc[month, "interest_rate"], 2)
            agent.savings = round(agent.savings + agent.post_tax_income - actual_consumption_cost + interest, 2)
            remaining_inventory = round(remaining_inventory - actual_consumption, 2)

        self.metrics.loc[month, "inventory"] = remaining_inventory

    def update_interest_rate(self, month):
        r_n = 0.02
        pi_target = 0.02
        alpha_pi = 0.5
        alpha_u = 0.5
        current_inflation = self.metrics.loc[month, "inflation"] if month > 0 else 0.0
        current_unemployment = self.metrics.loc[month, "unemployment"]
        interest_rate = r_n + current_inflation + alpha_pi * (current_inflation - pi_target) + alpha_u * (pi_target - current_unemployment)
        interest_rate = max(interest_rate, 0.0)
        self.metrics.loc[month, "interest_rate"] = round(interest_rate, 4)

    def calculate_macroeconomic_metrics(self, month, h_agents):
        if month > 0:
            prev_price = self.metrics.loc[month - 1, "price"]
            current_price = self.metrics.loc[month, "price"]
            inflation = (current_price - prev_price) / prev_price if prev_price > 0 else 0.0
            self.metrics.loc[month, "inflation"] = round(inflation, 4)
        else:
            self.metrics.loc[month, "inflation"] = 0.0

        total_unemployment = sum(1 - agent.p_w for agent in h_agents)
        unemployment_rate = total_unemployment / len(h_agents) if len(h_agents) > 0 else 0.0
        self.metrics.loc[month, "unemployment"] = round(unemployment_rate, 4)

        wealths = [agent.savings for agent in h_agents]
        gini_coeff = self.calculate_gini(wealths) if np.var(wealths) > 0 else 0.0
        equality = (1 - gini_coeff) * (len(h_agents) - 1) / len(h_agents) if len(h_agents) > 0 else 0.0
        self.metrics.loc[month, "equality"] = round(equality, 4)

        avg_wealth = sum(wealths) / len(h_agents) if len(h_agents) > 0 else 0.0
        self.metrics.loc[month, "productivity"] = round(avg_wealth, 2)

    def update_wage_and_price(self, month, h_agents):
        if month == 0:
            return
        prev_inventory = self.metrics.loc[month - 1, "inventory"]
        total_demand_prev = sum((agent.p_c * agent.savings) / (self.metrics.loc[month - 1, "price"] if self.metrics.loc[month - 1, "price"] > 0 else 1) for agent in h_agents)
        if max(total_demand_prev, prev_inventory) == 0:
            phi = 0.0
        else:
            phi = (total_demand_prev - prev_inventory) / max(total_demand_prev, prev_inventory)

        alpha_w = 0.05
        wage_adjustment = random.uniform(0, alpha_w * abs(phi))
        self.current_wage *= (1 + wage_adjustment * np.sign(phi))
        self.current_wage = round(self.current_wage, 2)

        alpha_p = 0.03
        price_adjustment = random.uniform(0, alpha_p * abs(phi))
        new_price = self.metrics.loc[month - 1, "price"] * (1 + price_adjustment * np.sign(phi))
        self.metrics.loc[month, "price"] = round(new_price, 2)

    @staticmethod
    def calculate_gini(wealths):
        arr = np.array(wealths, dtype=float)
        arr = arr[arr >= 0]
        n = len(arr)
        if n == 0:
            return 0.0
        sorted_wealths = np.sort(arr)
        cumulative = np.cumsum(sorted_wealths)
        total = cumulative[-1]
        if total == 0:
            return 0.0
        gini = (n + 1 - 2 * np.sum(cumulative) / total) / n
        return round(gini, 4)