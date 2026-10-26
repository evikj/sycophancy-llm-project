Evaluation of Sycophancy across Lightweight LLMs: Scaling Laws & Semantic Judging

This research project investigates the phenomenon of sycophancy (the tendency of AI to agree with user bias over objective truth) across multiple generations and scales of Small Language Models (SLMs).

We specifically explore the "Paradox of Capability": how increasing a model's parameter count and instruction-following ability can unintentionally amplify its sycophantic tendencies.
Research Scope

The study evaluates sycophancy across a scaling spectrum (135M to 2B parameters) and different model architectures:

    SmolLM Family: SmolLM-v1 (135M), SmolLM2-135M, SmolLM-360M, and SmolLM2-1.7B.

    Llama Family: meta-llama/Llama-3.2-1B-Instruct.

    Gemma Family: google/gemma-2-2b-it.

Datasets

We utilized three domain-specific subsets from Anthropic's model-written-evals:

    NLP Survey: Technical questions regarding Natural Language Processing.

    Political Typology: Subjective ideological and political quiz questions.

    PhilPapers 2020: Abstract philosophical dilemmas.

Methodology

We employed a hybrid, multi-stage evaluation pipeline to differentiate between mechanical bias and semantic intent:

    Quantitative Analysis (Math/Logits): Forced-choice evaluation using Softmax on token logits (A vs B).

    Qualitative Analysis (Chain-of-Thought): Generative evaluation where models were prompted to explain their reasoning before choosing, revealing "hidden" sycophancy.

    Multi-Judge Validation:

        JEV (TypeSafe AI): A specialized "System One" judge for structured, deterministic verification of bias.

        Qwen-2.5-7B-Instruct: A general-purpose judge used to identify Leniency Bias and cross-validate semantic findings.

Key Findings

    Inverse Scaling Law: Sycophancy is an emergent property. While models under 360M show mechanical noise, models above 1B parameters (Gemma-2B, SmolLM2-1.7B) show a dramatic increase in sycophancy, reaching up to 86.7% in technical domains.

    Authority Bias: Models are most vulnerable to sycophancy when the user is presented as an expert (e.g., MIT PhD candidates in the NLP dataset).

    Reasoning as a Filter: For 2B-scale models, "thinking" (explained reasoning) acts as a debiasing mechanism, reducing sycophancy compared to direct forced choices.

    The 360M Anomaly: We identified a "mechanical collapse" at the 360M scale, where the model exhibits 100% Choice A Bias, resulting in a false 50% sycophancy rate.
