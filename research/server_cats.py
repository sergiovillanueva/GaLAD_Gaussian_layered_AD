"""Category lists for the server runs (interleaved by benchmark so partial results stay balanced)."""

MVTEC = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw",
         "tile", "toothbrush", "transistor", "wood", "zipper"]
VISA = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3",
        "pcb4", "pipe_fryum"]
LOCO = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
AUTOVI = ["engine_wiring", "pipe_clip", "pipe_staple", "tank_screw", "underbody_pipes", "underbody_screw"]
BTAD = ["01", "02", "03"]
GOODS = ["cigarette_box", "drink_bottle", "drink_can", "food_bottle", "food_box", "food_package"]
GROUPS = [("mvtec_AD", MVTEC), ("VisA", VISA), ("mvtec_loco_AD", LOCO), ("AutoVI", AUTOVI), ("btad", BTAD),
          ("GoodsAD", GOODS)]
ALL = [f"{b}/{c}" for i in range(15) for b, cats in GROUPS if i < len(cats) for c in [cats[i]]]
MASKED = [f"mvtec_AD/{c}" for c in ("capsule", "hazelnut", "pill", "screw", "toothbrush")] + [f"VisA/{c}" for c in VISA]
BENCH = ["mvtec_AD/bottle", "VisA/candle", "AutoVI/engine_wiring"]

if __name__ == "__main__":
    import sys
    print(" ".join({"all": ALL, "masked": MASKED, "bench": BENCH}[sys.argv[1]]))
