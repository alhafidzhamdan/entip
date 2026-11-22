# Step 3c: mitotic figure counting

[← step 3](step3-high-grade.md)

**Status: first finder and reference-set runs are built but not yet run ([`highgrade_mitosis/`](../../training/runs/highgrade_mitosis/README.md)).** No trained detector exists. Step 3 makes no `mitotic_activity` record, so step 6 asks for that
finding and leaves grade 2 versus 3 to the pathologist.

## What it does

Counts mitotic figures. It does not diagnose, grade or apply a threshold.

Output for a slide or any chosen area: the count, a 95% range, the assessable tissue in mm², a density map, and one
point per mitosis.

## What a correct count needs

- **A denominator:** the area of tissue that could actually be assessed.
- **Few misses:** true mitoses found.
- **Few false counts:** most tissue has no mitosis, so even a low false rate can swamp a small true count.
- **One count per dividing cell:** a telophase pair counts once.

## Proposed pipeline

1. Assessable-tissue mask, which gives the area.
2. Finder: a point-heatmap network at about 0.25 µm per pixel, tuned to miss very little.
3. Checker: a high-power crop with context, mitosis against pyknosis, apoptosis, karyorrhexis, lymphocyte or crush.
4. Merge duplicates.
5. Calibrated table of points, from which any count per area is a query.

## Proposed first run: a reference set

Lock the test patients first. Then, for 20 other patients with 40× scans, a computer picks 4 random 0.5 mm² tissue
fields each (40 mm² in total), with no model involved. A pathologist marks every mitosis, blind to diagnosis and any
model, as mitosis, look-alike or uncertain. A second reader repeats a subset, and the counts are frozen. Any detector
is then scored on recall, false counts per mm², and predicted against true count per field.

## Rules

- Models or labels with non-commercial terms never train the counter or choose cells for labelling.
- Split by patient, freeze the model before testing, and do not set cut-offs on test data.
