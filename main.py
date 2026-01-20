from simulation import Simulation
import config

def main():
    print("Running paper-standard simulation...")
    print(f"N={config.NUM_HOUSEHOLDS}, P={config.SIMULATION_MONTHS}")
    print(f"LLM model: {config.LLM_MODEL}")

    sim = Simulation()
    results = sim.run_full_simulation()

    print("\n=== Long-term Social Outcome (month >= 40) ===")
    for tax_system, df in results.items():
        long_term_df = df[df["month"] >= 40]
        social_outcome = (long_term_df["equality"] * long_term_df["productivity"]).mean()
        print(f"{tax_system:12s}: {social_outcome:.4f}")

if __name__ == "__main__":
    main()
