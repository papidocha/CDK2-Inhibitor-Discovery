# Automated Kinase Inhibition Assays (ADP-Glo)

This module contains the Python API protocols and custom labware definitions used to execute miniaturized 384-well ADP-Glo biochemical kinase assays on the Opentrons Flex platform.

## Hardware & Environment
* **Platform:** Opentrons Flex (API Version 2.22)
* **Pipettes:** Flex 8-Channel 50 µL (`flex_8channel_50`)
* **Tips:** Opentrons Flex 96 Tip Rack 50 µL

## Custom Labware Definitions
Precise sub-microliter dispensing is critical for the low-volume assay format. The following custom `.json` labware definitions are provided to ensure accurate bottom-clearance heights and prevent droplet retention:
* `corning3825_384_wellplate_50ul.json`: Destination assay plate (Corning 384-well, flat bottom, low-volume).
* `agilent_96_wellplate_330ul.json`: Source plate for compound libraries and enzyme/substrate master mixes.

*(Note: Load these `.json` files into your Opentrons App via the Labware Management tab prior to execution).*

## Protocol: `DuoTip_FlexibleAnchor_20Inhib_dupl_48Wells_v7.py`
This script automates the transfer and layout of 20 distinct CDK2 inhibitors, executed in duplicate across a 48-well block mapping within the primary 384-well assay plate. It includes a custom `smart_transfer` helper function to handle 2 µL buffer limits and dynamic Z-height dispensing.

### Flex Deck Layout
| Deck Slot | Labware | Purpose |
| :---: | :--- | :--- |
| **A1** | Corning 384-well Plate (`corning3825_384_wellplate_50ul`) | Destination assay plate (reaction execution) |
| **A2** | Agilent 96-well Plate (`agilent_96_wellplate_330ul`) | Source plate (inhibitors, kinase, ATP, detection) |
| **A3** | Opentrons Flex 50 µL Tip Rack | Multi-channel transfers |
| **D1** | Trash Bin | Protocol waste |
