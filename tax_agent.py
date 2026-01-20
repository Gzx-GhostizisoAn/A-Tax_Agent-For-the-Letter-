import json
import config
from config import TAX_BRACKETS, NUM_HOUSEHOLDS
from llm_client import call_llm_json


class TaxAgent:
    def __init__(self):
        self.tax_history = []
        self.theta_G = {"target_equality": 0.7, "target_productivity": 50000}
        self.theta_H = {"avg_hh_income": 0.0}

    def adjust_tax_rates(self, month, env, h_agents):
   
        current_eq = env.metrics.loc[month - 1, "equality"] if month > 0 else 0.0
        current_prod = env.metrics.loc[month - 1, "productivity"] if month > 0 else 0.0
        hh_incomes = [agent.pre_tax_income for agent in h_agents]
        self.theta_H["avg_hh_income"] = round(sum(hh_incomes) / NUM_HOUSEHOLDS, 2) if NUM_HOUSEHOLDS > 0 else 0.0

        # If LLM enabled, attempt to call it for tax rates
        if getattr(config, 'LLM_ENABLED', False):
            try:
                prompt = f"""
                You are a tax planner in charge of adjusting tax rates for 7 income brackets: {TAX_BRACKETS}.
                Last month's key metrics:
                - Average household income: ${self.theta_H['avg_hh_income']:.2f}
                - Equality (1 - normalized Gini): {current_eq:.4f}
                - Average productivity (avg wealth): ${current_prod:.2f}
                - Household incomes: {[agent.pre_tax_income for agent in h_agents]}
                - Household wealths: {[agent.savings for agent in h_agents]}

                Provide ONLY a list of 7 tax rates (JSON format). No other content!
                Example: [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]
                """
                tax_rates = call_llm_json(prompt, model=getattr(config, 'LLM_MODEL', None))
                if isinstance(tax_rates, list) and len(tax_rates) >= len(TAX_BRACKETS):
                    tax_rates = [min(max(round(float(r), 2), 0.0), 0.99) for r in tax_rates[:len(TAX_BRACKETS)]]
                    self.tax_history.append({"month": month, "rates": tax_rates})
                    env.metrics.at[month, "tax_rates"] = tax_rates
                    # update theta_G heuristics as before
                    ideal_equality = 0.75
                    ideal_productivity = 60000
                    self.theta_G["target_equality"] = round((current_eq + ideal_equality) / 2, 4)
                    self.theta_G["target_productivity"] = round((current_prod + ideal_productivity) / 2, 2)
                    return tax_rates
            except Exception:
                pass


        base_rates = [0.08, 0.12, 0.20, 0.24, 0.30, 0.35, 0.40]
        eq_factor = (0.5 - current_eq)
        prod_factor = (current_prod - 50000) / (50000 + 1)
        new_rates = []
        for i, r in enumerate(base_rates):
            if i >= len(base_rates) - 2:
                adj = 0.2 * eq_factor - 0.1 * prod_factor
            else:
                adj = 0.05 * eq_factor - 0.02 * prod_factor
            rate = min(max(round(r + adj, 2), 0.0), 0.99)
            new_rates.append(rate)

        self.tax_history.append({"month": month, "rates": new_rates})
        env.metrics.at[month, "tax_rates"] = new_rates
        return new_rates