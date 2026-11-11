"""Step 4 of the AI glioma workflow: predict the CNS methylation class from an H&E slide.

Research use only. The output is a suggestion with a confidence; when the model is not confident it says
so and asks for the methylation array (``order_test``). See ``training/methylation.md``.
"""
