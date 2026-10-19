from opentrons import protocol_api

metadata = {
    'protocolName': 'CDK Duo-Tip - 20 Inhibitors',
    'author': 'Sven M. Papidocha'
}

requirements = {"robotType": "Flex", "apiLevel": "2.22"}

# --- CONFIGURATION ---
START_WELL = 'A1'  # Anchor for the 12x4 block (e.g., 'A1' or 'H13')
# ---------------------

def run(protocol: protocol_api.ProtocolContext):
    # --- 1. Anchor & Bounds Logic ---
    anchor_row_char = START_WELL[0].upper()
    anchor_col = int(START_WELL[1:])

    if not ('A' <= anchor_row_char <= 'M') or not (1 <= anchor_col <= 13):
        raise ValueError(f"START_WELL {START_WELL} out of bounds.")

    r1 = anchor_row_char
    r2 = chr(ord(anchor_row_char) + 1)  # Duplicate of r1

    cols = [anchor_col + i for i in range(12)]
    blank_dmso_col = cols[11]
    inhib_kinase_cols = cols[:11]

    # --- 2. Deck Layout ---
    trash = protocol.load_trash_bin("D1")
    dest_plate = protocol.load_labware('corning3825_384_wellplate_50ul', 'A1')
    source_plate = protocol.load_labware('agilent_96_wellplate_330ul', 'A2')
    tiprack = protocol.load_labware('opentrons_flex_96_tiprack_50ul', 'A3')
    p50_multi = protocol.load_instrument('flex_8channel_50', 'left')

    # --- 3. Smart Transfer Helper (Trash Buffer + Auto Touch Tip) ---
    def smart_transfer(source_well, dispense_plan, touch=False):
        # Cap at 48uL chunk + 2uL buffer = 50uL perfect aspiration
        MAX_VOL = 48 
        remaining = dispense_plan.copy()
        
        while remaining:
            chunk_vol = 0
            chunk = []
            while remaining and (chunk_vol + remaining[0][0] <= MAX_VOL):
                vol, loc = remaining.pop(0)
                chunk.append((vol, loc))
                chunk_vol += vol
                
            # THE BULLETPROOF FAILSAFE: Stops the script cleanly, no hardware crash!
            if not chunk:
                oversized_vol = remaining[0][0]
                raise ValueError(
                    f"Protocol Error: Tried to dispense {oversized_vol} uL. "
                    f"This exceeds the {MAX_VOL} uL safe limit for 50 uL tips with a buffer."
                )
                
            # Aspirate the reaction volume PLUS a 2uL cushion
            p50_multi.aspirate(chunk_vol + 2, source_well)
            
            for vol, loc in chunk:
                p50_multi.dispense(vol, loc)
                
                # Automatically touch the tip to the wall of the current well
                if touch:
                    p50_multi.touch_tip()
                
            # Dispense the 2uL cushion into the trash to reset internal volume tracker
            p50_multi.dispense(2, trash.top())

    # --- 4. Reagent Mapping ---
    dmso_anchor = source_plate['A11']
    blank_anchor = source_plate['A12']
    kinase_anchor = source_plate['C1']
    atp_anchor = source_plate['C2']
    adp_glo_anchor = source_plate['C3']
    det_primary_anchor = source_plate['C4']
    det_duplicate_anchor = source_plate['C5']

    # --- 5. Tip Tracking ---
    duo_tips = [tiprack[f'{r}{c}'] for c in range(1, 13) for r in ['G', 'E', 'C', 'A']]
    tip_counter = 0

    def pick_up_duo():
        nonlocal tip_counter
        p50_multi.pick_up_tip(duo_tips[tip_counter])
        tip_counter += 1

    # --- PHASE 1: KINASE & BLANK (Z = 0.25mm) ---
    p50_multi.flow_rate.aspirate, p50_multi.flow_rate.dispense = 5, 5

    # Blank (Using touch=True to knock off droplets)
    pick_up_duo()
    smart_transfer(blank_anchor.bottom(0.5), [
        (4, dest_plate[f'{r1}{blank_dmso_col}'].bottom(0.25)),
        (4, dest_plate[f'{r2}{blank_dmso_col}'].bottom(0.25))
    ], touch=True)
    p50_multi.drop_tip()

    # Kinase (Using touch=True to knock off droplets)
    pick_up_duo()
    dispenses = []
    for col in reversed(inhib_kinase_cols):
        dispenses.append((4, dest_plate[f'{r1}{col}'].bottom(0.25)))
        dispenses.append((4, dest_plate[f'{r2}{col}'].bottom(0.25)))
    smart_transfer(kinase_anchor.bottom(0.5), dispenses, touch=True)
    p50_multi.drop_tip()

    protocol.pause("Phase 1 Complete.")

    # --- PHASE 2: INHIBITORS (Z = 0.4mm) ---
    pick_up_duo()
    smart_transfer(dmso_anchor.bottom(0.5), [
        (2, dest_plate[f'{r1}{blank_dmso_col}'].bottom(0.4)),
        (2, dest_plate[f'{r2}{blank_dmso_col}'].bottom(0.4)),
        (2, dest_plate[f'{r1}{inhib_kinase_cols[-1]}'].bottom(0.4)),
        (2, dest_plate[f'{r2}{inhib_kinase_cols[-1]}'].bottom(0.4))
    ])
    p50_multi.drop_tip()

    for i, col in enumerate(reversed(inhib_kinase_cols)):
        if i == 0: continue
        pick_up_duo()
        smart_transfer(source_plate[f'A{11 - i}'].bottom(0.4), [
            (2, dest_plate[f'{r1}{col}'].bottom(0.4)),
            (2, dest_plate[f'{r2}{col}'].bottom(0.4))
        ])
        p50_multi.drop_tip()

    protocol.pause("Phase 2 Complete.")

    # --- PHASE 3: ATP (Z = 0.8mm) ---
    pick_up_duo()
    dispenses = []
    for col in reversed(cols):
        dispenses.append((4, dest_plate[f'{r1}{col}'].bottom(0.8)))
        dispenses.append((4, dest_plate[f'{r2}{col}'].bottom(0.8)))
    smart_transfer(atp_anchor.bottom(0.5), dispenses)
    p50_multi.drop_tip()

    protocol.pause("Reaction Started.")

    # --- PHASE 4: ADP-GLO (Z = 4.0mm) ---
    p50_multi.flow_rate.aspirate, p50_multi.flow_rate.dispense = 10, 10
    pick_up_duo()
    for start_col in [12, 8, 4]:
        dispenses = []
        for i in range(4):
            col = cols[start_col - anchor_col - i]
            dispenses.append((10, dest_plate[f'{r1}{col}'].bottom(4.0)))
            dispenses.append((10, dest_plate[f'{r2}{col}'].bottom(4.0)))
        smart_transfer(adp_glo_anchor.bottom(0.5), dispenses)
    p50_multi.drop_tip()

    protocol.pause("ADP-Glo Added.")

    # --- PHASE 5: DETECTION (Z = 6.0mm) ---
    pick_up_duo()
    for i in range(12, 0, -2):
        col1, col2 = cols[i - 1], cols[i - 2]
        # Primary Set
        smart_transfer(det_primary_anchor.bottom(0.5), [
            (20, dest_plate[f'{r1}{col1}'].bottom(6.0)),
            (20, dest_plate[f'{r1}{col2}'].bottom(6.0))
        ])
        # Duplicate Set
        smart_transfer(det_duplicate_anchor.bottom(0.5), [
            (20, dest_plate[f'{r2}{col1}'].bottom(6.0)),
            (20, dest_plate[f'{r2}{col2}'].bottom(6.0))
        ])
    p50_multi.drop_tip()

    protocol.comment("Quad-Row Complete (Aspirate = Dispense).")