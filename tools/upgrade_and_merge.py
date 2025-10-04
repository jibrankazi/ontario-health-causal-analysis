# bsts.json is optional
bsts = {"att": None, "ci": [None, None], "p": None, "relative_effect": None, "notes": "BSTS not run"}
if bsts_p.exists():
    try:
        b = json.loads(bsts_p.read_text())

        # Coerce p -> float or None
        raw_p = b.get("p")
        p_val = None
        if raw_p is not None:
            try:
                # handle "NA", "NaN", "", "null" gracefully
                if str(raw_p).strip().lower() not in ("na", "nan", "null", ""):
                    p_val = float(raw_p)
            except Exception:
                p_val = None

        # Coerce ci -> [float|None, float|None]
        ci = b.get("ci") or [None, None]
        if not isinstance(ci, (list, tuple)) or len(ci) < 2:
            ci = [None, None]
        ci0 = None if ci[0] is None or str(ci[0]).strip().lower() in ("na","nan","null","") else float(ci[0])
        ci1 = None if ci[1] is None or str(ci[1]).strip().lower() in ("na","nan","null","") else float(ci[1])

        # Coerce relative_effect -> float or None
        re = b.get("relative_effect")
        try:
            re = float(re)
        except Exception:
            re = None

        bsts = {
            "att": b.get("att"),
            "ci": [ci0, ci1],
            "p": p_val,
            "relative_effect": re,
            "notes": b.get("notes"),
        }
    except Exception as e:
        bsts["notes"] = f"Failed to read bsts.json: {e}"
