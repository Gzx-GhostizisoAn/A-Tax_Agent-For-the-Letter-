import matplotlib.pyplot as plt
import config
from config import NUM_HOUSEHOLDS, SIMULATION_MONTHS
from macro_env import MacroeconomicEnvironment 
from h_agent import HAgent  
from tax_agent import TaxAgent  

class Simulation:
    def __init__(self):
        self.env = MacroeconomicEnvironment()
        self.h_agents = [HAgent(i) for i in range(NUM_HOUSEHOLDS)]
        self.tax_agent = TaxAgent()

        self.baseline_tax_rates = {
            "us_federal": [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37],  
            "saez": [0.15, 0.20, 0.25, 0.30, 0.38, 0.45, 0.50],      
            "free_market": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]  
        }

    def run_single_month(self, month, tax_system="tax_agent"):
  
        if tax_system == "tax_agent":
            if month == 0:
              
                initial_rates = self.baseline_tax_rates["us_federal"]
                self.env.metrics.at[month, "tax_rates"] = initial_rates
                self.tax_agent.tax_history.append({"month": 0, "rates": initial_rates})
            else:
          
                self.tax_agent.adjust_tax_rates(month, self.env, self.h_agents)
        else:
           
            self.env.metrics.at[month, "tax_rates"] = self.baseline_tax_rates[tax_system]

    
        for agent in self.h_agents:
            agent.make_decision(month, self.env)


        total_labor = self.env.calculate_total_labor_supply(self.h_agents)
        self.env.update_inventory_after_production(month, total_labor)


        self.env.calculate_pre_tax_income(self.h_agents, total_labor)

    
        self.env.calculate_tax_and_redistribution(month, self.h_agents)


        self.env.update_consumption_and_inventory(month, self.h_agents)

      
        self.env.update_wage_and_price(month,self.h_agents)


        self.env.update_interest_rate(month)


        self.env.calculate_macroeconomic_metrics(month, self.h_agents)

 
        for agent in self.h_agents:
            agent.self_reflect(month)

    def run_full_simulation(self):
      
        results = {}
   
        for tax_system in ["tax_agent", "us_federal", "saez", "free_market"]:
            print(f"Running simulation for {tax_system}...")
           
            self.env = MacroeconomicEnvironment()
            self.h_agents = [HAgent(i) for i in range(NUM_HOUSEHOLDS)]
            self.tax_agent = TaxAgent()

          
            for month in range(SIMULATION_MONTHS):
                self.run_single_month(month, tax_system)

      
            results[tax_system] = self.env.metrics.copy()

        
        self.visualize_results(results)
        return results
    def self_reflect(self, month):
        
            if month % config.REFLECTION_INTERVAL != 0 or len(self.memo) < config.REFLECTION_INTERVAL:
                return

    
            recent_memo = self.memo[-config.REFLECTION_INTERVAL:]

            avg_p_w = round(
                sum(entry["p_w"] for entry in recent_memo) / config.REFLECTION_INTERVAL, 2
            )
            avg_p_c = round(
                sum(entry["p_c"] for entry in recent_memo) / config.REFLECTION_INTERVAL, 2
            )
        
            self.theta_R = {
                "avg_p_w": avg_p_w,
                "avg_p_c": avg_p_c
            }

       
            self.p_w = round((self.p_w + avg_p_w) / 2, 2)
            self.p_c = round((self.p_c + avg_p_c) / 2, 2)


    def visualize_results(self, results):
       
        plt.rcParams['font.sans-serif'] = ['Arial']
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    
        ax1 = axes[0, 0]
        for tax_system, df in results.items():
            equity_productivity = df["equality"] * df["productivity"]
            ax1.plot(df["month"], equity_productivity, label=tax_system.replace("_", " ").title())
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Equality-Productivity Index")
        ax1.set_title("Long-Term Social Outcomes (TaxAgent vs Baselines)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)


        ax2 = axes[0, 1]
        for tax_system, df in results.items():
            ax2.plot(df["month"], df["equality"], label=tax_system.replace("_", " ").title())
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Equality (1 - Normalized Gini)")
        ax2.set_title("Equality Over Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    
        ax3 = axes[1, 0]
        for tax_system, df in results.items():
            ax3.plot(df["month"], df["inflation"] * 100, label=tax_system.replace("_", " ").title())  # 转为百分比
        ax3.set_xlabel("Month")
        ax3.set_ylabel("Inflation Rate (%)")
        ax3.set_title("Inflation Over Time")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

       
        ax4 = axes[1, 1]
        for tax_system, df in results.items():
            ax4.plot(df["month"], df["unemployment"] * 100, label=tax_system.replace("_", " ").title())  # 转为百分比
        ax4.set_xlabel("Month")
        ax4.set_ylabel("Unemployment Rate (%)")
        ax4.set_title("Unemployment Over Time")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("tax_agent_simulation_results.png", dpi=300, bbox_inches="tight")
        plt.show()