# JEDI: Justifiable End-dialogue Driven Interaction for NPC Entities in Role-Playing Games
![image](https://github.com/Willy-Chan/JEDI/assets/106504264/af32894b-da5c-4c4d-aa70-13fdb829a26e)
![image](https://github.com/Willy-Chan/JEDI/assets/106504264/16a2524a-f574-430c-8ee7-bea9cd1040ac)

## Authors
- Willy Chan (willyc@stanford.edu)
- Omar Abul-Hassan (omarah@stanford.edu)
- Sokserey Sun (sokserey@stanford.edu)

## Abstract
In the realm of story-driven role-playing games (RPGs), generating dynamic, context-sensitive dialogue is pivotal for enhancing player immersion and engagement. Traditional methods rely on pre-written scripts, which are labor-intensive and limited in adaptability. JEDI investigates the potential of large language models to revolutionize RPG dialogue generation, using the game "Star Wars: Knights of the Old Republic" (KOTOR) as a case study. We preprocess the KOTOR dataset to graph and linearize dialogue sequences alongside game state information, then fine-tune state-of-the-art LLMs (BART, GPT-2, GPT-3.5 Turbo) to generate contextually relevant and engaging dialogue. Our evaluation, employing metrics like BLEURT, BERTScore, and DialogueRPT, demonstrates significant improvements in generating coherent and contextually appropriate dialogue when fine-tuning. The final fine-tuned BART model's BLEURT score improved from -1.3090 to -0.9215, while GPT-3.5 Turbo achieved a BERTScore of 0.8940. Additionally, GPT-3.5 Turbo's DialogRPT scores for human-vs-rand and human-vs-machine were 0.5118 and 0.999, respectively. The introduction of cross-attention in GPT-2 further enhanced its performance, with BLEURT improving from -0.8000 to -0.7014, and achieving a BERTScore of 0.8535. This work contributes a framework for integrating LLMs in branching, narrative-based RPGs, paving the way for more interactive and immersive game narratives.

## Introduction
Video games, particularly story-driven RPGs, are expanding rapidly in the entertainment industry. RPGs feature complex narratives, allowing players to shape the game's outcome through their choices. Traditional dialogue generation in RPGs relies on pre-written scripts, which are resource-intensive. Large language models (LLMs) present an opportunity to generate dynamic, context-sensitive dialogue. This project focuses on developing an end-to-end dialogue generation system for RPGs, using the game "Star Wars: Knights of the Old Republic" (KOTOR) as a case study. We preprocess the KOTOR dataset to graph and linearize dialogue sequences alongside game state information, then fine-tune state-of-the-art LLMs (BART, GPT-2, GPT-3.5 Turbo) to generate contextually relevant and engaging dialogue.

## Related Work
Previous work by Akoury et al. explores methods of generating relevant datasets using the game "Disco Elysium." They structured datasets as graphs and linearized them into Lua scripts, using clustering techniques to group similar dialogue nodes. Other studies highlight the effectiveness of fine-tuning transformer models for generating RPG item descriptions and sustaining long-form conversations. Inspired by these works, we applied similar techniques to the KOTOR dataset, adapting the clustering and linearization approach for our dialogue sequences.

## Approach
We fine-tuned three models: BART, GPT-2, and GPT-3.5 Turbo. The models were fine-tuned using sequences from the KOTOR dataset, which were graphically represented and linearized. For GPT-2, we added a cross-attention mechanism to improve context sensitivity. Training involved masking dialogue portions and providing the models with context from preceding and succeeding text.

### Models:
- **BART**: Bidirectional and Auto-Regressive Transformers, employing an encoder-decoder architecture with a denoising objective.
- **GPT-2**: Unidirectional transformer with multi-head self-attention, fine-tuned with cross-attention to incorporate game state information.
- **GPT-3.5 Turbo**: Enhanced contextual understanding and few-shot learning capabilities with significantly more parameters than GPT-2.

## Experiments
### Data
The dataset includes over 29,000 unique interactions from KOTOR, structured as a graph with nodes representing dialogue and game state information. Dialogue sequences were linearized, and random dialogue options were masked to provide context for the models.

### Evaluation
We employed BLEURT, BERTScore, DialogueRPT, BLEU, and ROUGE metrics to evaluate model performance. These metrics assess semantic similarity, grammatical correctness, contextual appropriateness, and human-likeness of the generated dialogue.

## Results
- **BART**: Significant improvements in BLEURT and BERTScore, modest improvements in DialogRPT scores.
- **GPT-2 with Cross-Attention**: Enhanced performance across all metrics compared to standard GPT-2, leveraging game state information effectively.
- **GPT-3.5 Turbo**: Best overall performance, achieving high BERTScore and BLEURT, demonstrating strong contextual and semantic understanding.

### Evaluation Metrics
| Metric                          | bart-base Default | bart-base Fine-Tuned | GPT-2 | GPT2CA | GPT-3.5 Fine-Tuned |
|---------------------------------|-------------------|----------------------|-------|--------|---------------------|
| DialogRPT (human-vs-rand)       | 0.5030            | 0.5130               | 0.5090| 0.5105 | 0.5118              |
| DialogRPT (human-vs-machine)    | 0.9980            | 0.9970               | 0.9985| 0.9987 | 0.9990              |
| BERTScore (avg F1 Score)        | 0.8295            | 0.8602               | 0.8500| 0.8535 | 0.8940              |
| BLEURT Score                    | -1.3090           | -0.9215              | -0.8000| -0.7014| -0.5579             |
| Validation Loss                 | 4.8130            | 1.8798               | 2.1000| 1.9502 | 1.0075              |
| BLEU Score                      | 0.0056            | 0.0458               | 0.0400| 0.0420 | 0.1847              |
| ROUGE-1 F-Measure               | 0.1580            | 0.2885               | 0.2700| 0.2780 | 0.3589              |
| ROUGE-2 F-Measure               | 0.0339            | 0.2225               | 0.2100| 0.2150 | 0.2563              |
| ROUGE-L F-Measure               | 0.1358            | 0.2785               | 0.2650| 0.2774 | 0.3370              |
