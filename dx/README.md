# Integrated diagnosis (AI glioma workflow, step 5)

A fixed WHO CNS5 rules engine. It reads the finding records from steps 1 to 4 (and real test results),
works out the integrated diagnosis, orders real tests for anything uncertain, and writes a draft layered report
for the pathologist to sign off. Research use only.

The shared contract for steps 1 to 4 is in [`contract/`](contract/README.md).
