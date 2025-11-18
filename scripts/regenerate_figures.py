def main():
    if not RES.exists():
        raise SystemExit("results/results.json not found. Run: python src/run_analysis.py")
    
    r = json.loads(RES.read_text())
    
    # ←←← FIXED: Now reads nested structure correctly
    did  = r["did"].get("att")
    psm  = r["psm"].get("att")
    bsts = r["bsts"].get("att")
    # →→→ end of fix

    labels = ["DiD", "PSM", "BSTS"]
    values = [0 if v is None else float(v) for v in [did, psm, bsts]]

    FIG.mkdir(parents=True, exist_ok=True)
    s = pd.Series(values, index=labels)
    ax = s.plot(kind="bar", color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Estimated Treatment Effects on the Treated (ATT)")
    ax.set_ylabel("Change in Incidence per 100,000")
    ax.axhline(0, color='black', linewidth=0.8)

    for i, (label, val) in enumerate(zip(labels, [did, psm, bsts])):
        display_val = "NA" if val is None else f"{val:.1f}"
        ax.text(i, val + (1 if val > 0 else -2), display_val, 
                ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIG / "att_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure saved: figures/att_summary.png")
