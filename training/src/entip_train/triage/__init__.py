"""Step 1 of the AI glioma workflow: is this tissue tumour, and is the tumour glial?

One attention-MIL model over H-optimus-0 tile bags gives six classes; tumour and glial probabilities are
sums of those. Every slide either gets a confident call or is sent to a pathologist. Research use only.
"""
