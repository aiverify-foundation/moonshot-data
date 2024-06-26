<div align="center">

![Moonshot Logo](https://github.com/moonshot-admin/moonshot/raw/main/misc/aiverify-moonshot-logo.png)

This repository contains the test assets needed for [Project Moonshot](https://github.com/aiverify-foundation/moonshot)

[![Python 3.11](https://img.shields.io/badge/python-3.11-green)](https://www.python.org/downloads/release/python-3111/)


</div>

<b>Motivation </b>

Developed by the [AI Verify Foundation](https://aiverifyfoundation.sg/?utm_source=Github&utm_medium=referral&utm_campaign=20230607_AI_Verify_Foundation_GitHub), [Moonshot](https://aiverifyfoundation.sg/project-moonshot/?utm_source=Github&utm_medium=referral&utm_campaign=20230607_Queries_from_GitHub) is one of the first tools to bring benchmarking and red teaming together to help AI developers, compliance teams and AI system owners <b>evaluate LLMs and LLM applications</b>.
This repository contains the test assests intended to work with the Moonshot Library. You can also [contribute](#contributing) to Project Moonshot's testing capabilities.

Go to [Project Moonshot Repository](https://github.com/aiverify-foundation/moonshot).

</br>

## Table of Contents

#### 🔗 For accessing AI systems:
- [Connectors](#connectors) are APIs to AI systems to be tested and AI systems that enable scoring metrics, attack modules and context strategies.
- <b>Connector Endpoints</b> are ready-to-use connectors that have been configured with the necessary API tokens and parameters.

#### 📊 For Benchmarking:
- <b>Datasets</b> are a collection of input-target pairs, where the 'input' is a prompt provided to the AI system being tested, and the 'target' is the correct response (if any). 
- <b>Metrics</b> are predefined criteria used to evaluate the LLM’s outputs against the targets defined in the recipe's dataset. These metrics may include measures of accuracy, precision, or the relevance of the LLM’s responses.
- <b>Prompt Templates</b> are predefined text structures that guide the formatting and contextualisation of inputs in recipe datasets. Inputs are fit into these templates before being sent to the AI system being tested.
- [Recipes](#recipes) are a benchmarks that are ready to be administered onto an AI system, consisting minimally of a dataset and a metric.
- [Cookbooks](#cookbooks) are thematic sets of recipes that are ready to be administered onto an AI system.

#### ☠️ For Red Teaming:
- [Attack Modules](#attack-modules) are techniques that will enable the automatic generation of adversarial prompts for automated red teaming.
- <b>Context Strategies</b> are predefined approaches to append the red teaming session's context to each prompt.
- <b>Prompt Templates</b> are predefined text structures that guide the formatting and contextualisation of the prompt sent to the AI system being tested. User-input prompts are fit into these templates before being sent to the AI system being tested.

#### 💯 Results:
- <b>Generated Outputs</b> directory contains files that are automatically produced when tests are run. There are mainly three types of files:
    - <b>Databases</b> directory contains DB files that are generated when a runner is created. It contains information related to benchmark runs and red teaming sessions. This include details such as the prompts used, the predictions made by the LLMs, and the time taken for these predictions. 
    - <b>Results</b> directory contains JSON files that hold the results of the benchmark runs, which have been formatted and processed by the selected Results Modules
    - <b>Runners</b> directory contains JSON files that store metadata information, such as the location of the database file, which holds the records of the results. 

- <b>Results Modules</b> directory contains modules that format the raw results that are generated from the benchmark tests. 


#### 🤝 Enablers:
- <b>Database Modules</b> directory contains modules that allow us to connect to various databases, such as SQLite. 
- <b>I/O Modules</b> directory contains modules that allow us to read and writing operations for data handling, such as JSON.
- <b>Runner Modules</b> directory contains modules that help us run benchmarking tests and red teaming sessions.


</br>

## Getting Started

### ✅ Prerequisites
1. [Python 3.11](https://www.python.org/downloads/) (We have yet to test on later releases)

2. [Git](https://github.com/git-guides/install-git)

3. [Moonshot](https://github.com/aiverify-foundation/moonshot)

### ⬇️ Installation

Run the following command

```
python -m moonshot -i moonshot-data
```

## Contributing

Any contributions are greatly appreciated.

Please fork the repo and create a pull request. You can also open an issue with the tag "enhancement". Do give the project a star too!

1. Fork the `moonshot-data` Project
2. Install `moonshot` (to run your test assets)
3. Create your branch (`git checkout -b metric/X` or `git checkout -b cookbook/X` or `git checkout -b recipe/X` or )
4. Push to the branch (`git push origin metric/X`)
5. Open a Pull Request

## Current Collection

*Last Updated 28 May*

### Attack Modules  

| Attack Modules | description |  
|---|---|   
| Charswap Attack |  This module tests for adversarial textual robustness. It creates perturbations through swapping characters for words that contains more than 3 characters.&lt;br&gt;Parameters:&lt;br&gt;1. MAX_ITERATIONS - Number of prompts that should be sent to the target. [Default: 10]&lt;br&gt;2. word_swap_ratio - Percentage of words in a prompt that should be perturbed. [Default: 0.2]&lt;br&gt; |  
| Colloquial Wordswap Attack | This attack module tests for textual robustness against the Singapore context. It takes in prompts that feature nouns that describe people. Examples of this include words like &#x27;girl&#x27; , &#x27;boy&#x27; or &#x27;grandmother&#x27;. The module substitutes these words with their Singapore colloquial counterparts, such as &#x27;ah boy&#x27;, &#x27;ah girl&#x27; and &#x27;ah ma&#x27;. |  
| Homoglyph Attack |  This module tests for adversarial textual robustness. Homoglyphs are alternative words for words comprising of ASCII characters.&lt;br&gt;Example of a homoglyph fool -&gt; fooI&lt;br&gt;This module purturbs the prompt with all available homoglyphs for each word present.&lt;br&gt;Parameters:&lt;br&gt;1. MAX_ITERATIONS - Maximum number of prompts that should be sent to the target. [Default: 20]. |  
| Insert Punctuation Attack |  This module tests for adversarial textual robustness and creates perturbations through adding punctuation to the start of words in a prompt.&lt;br&gt;Parameters:&lt;br&gt;1. MAX_ITERATIONS - Number of prompts that should be sent to the target. [Default: 10]&lt;br&gt;2. word_swap_ratio - Percentage of words in a prompt that should be perturbed. [Default: 0.2]. |  
| Job Role Generator | This attack module adds demographic groups to the job role. |  
| Malicious Question Generator | This attack module generates malicious questions using OpenAI&#x27;s GPT4 based on a given topic. This module will stop by the number of iterations (Default: 50). To use this attack module, you need to configure an &#x27;openai-gpt4&#x27;endpoint. |  
| Sample Attack Module | This is a sample attack module. |  
| Textfooler | This module tests for adversarial textual robustness and implements the perturbations listed in the paper Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment. &lt;br&gt;Parameters:&lt;br&gt;1. MAX_ITERATIONS - Number of prompts that should be sent to the target. This is also the number of transformations that should be generated. [Default: 5]&lt;br&gt;2. word_swap_ratio - Percentage of words in a prompt that should be perturbed. [Default: 0.2]&lt;br&gt;3. cosine_sim - Minimum word embedding cosine similarity [Default: 0.5]&lt;br&gt;4. window_size - Window size for the Universal Sentence Encoder (USE). [Default: 15]&lt;br&gt;5. threshold - Semantic similarity threshold for the USE. [Default: 0.840845057]&lt;br&gt;6. max_candidates - Number of nearest candidates to swap words with. [Default: 50]&lt;br&gt;Note:&lt;br&gt;Usage of this attack module requires the internet. Initial downloading of the GLoVe embedding occurs when the UniversalEncoder is called.&lt;br&gt;Embedding is retrieved from the following URL: https://textattack.s3.amazonaws.com/word_embeddings/paragramcf |  
| Textbugger | This module tests for adversarial textual robustness and implements the perturbations listed in the paper: TEXTBUGGER: Generating Adversarial Text Against Real-world Applications.\nParameters:\n1. MAX_ITERATIONS - Number of prompts that should be sent to the target. This is also the number of transformations that should be generated. [Default: 5]<br>2. word_swap_ratio - Percentage of words in a prompt that should be perturbed. [Default: 0.2]<br>3. top_k - To select top k number of semantic words from the GLoVe embedding. [Default: 5]<br>4. threshold - semantic similarity threshold for the universal encoder. [Default: 0.8]<br>Note:<br>Usage of this attack module requires the internet. Initial downloading of the GLoVe embedding occurs when the UniversalEncoder is called.\nEmbedding is retrieved from the following URL: https://textattack.s3.amazonaws.com/word_embeddings/paragramcf" |  
| Toxic Sentence Generator |  This module generates toxic sentence based on a given seed prompt. The attack module intends to test if the system under tests will complete the sentence with toxic sentences/phrases. | 
| Violent Durian | This is a multi-turn agent designed to interact over several exchanges. It&#x27;s used to elicit dangerous or violent suggestions from the target language model by adopting a criminal persona. The application is experimental and uses OpenAI GPT-4. Configure the endpoint openai-gpt4 to use this attack module. |  

</br>

### Cookbooks

| Cookbook | description |  
|---|---|   
| AI Safety in Chinese Language | This cookbook measures the system&#x27;s ability in answering trust and safety questions that are asked in Chinese languages. |  
| Easy test sets for Common Risks | This is a cookbook that consists (easy) test sets for common risks. These test sets are adapted from various research and will be expanded in the future. |  
| Hard test sets for Common Risks | This is a cookbook that consists (hard) test sets for common risks. These test sets are adapted from various research and will be expanded in the future. |  
| Leaderboard Cookbook | This cookbook, drawing inspiration from leaderboards like HF&#x27;s OpenLLM and HELM, features popular benchmarks for testing model capabilities, with results that may vary from actual leaderboard standings. |  
| Legal Summarisation | This cookbook runs general capability benchmark on legal summarisation model. |  
| Medical LLM Leaderboard | This cookbook contains the benchmarks used in Medical LLM Leaderboard hosted on HuggingFace. Achieving a high score may mean that the targeted system is performing well in answering healthcare questions. |  
| MLCommons AI Safety Benchmarks v0.5 for General Chat Models | This cookbook systematically assesses the system&#x27;s capacity to navigate and respond to potentially harmful inquiries, analyzing its responses in accordance with the established safety guidelines. A high refusal rate means that the system under test is performing very well. A low refusal rate means that the system under test is unsafe. |  
| Facts about Singapore | This cookbook is designed to evaluate Singapore&#x27;s historical events and essential facts, serving as a litmus test for its understanding of the country&#x27;s unique context. In addition, there are safety prompts written in Singapore context. By assessing a model&#x27;s familiarity with Singapore&#x27;s cultural and historical landscape, it provides valuable insights into its overall proficiency and accuracy in natural language processing systems tailored to Singaporean contexts. |  
| Tamil Language | This is a cookbook that consists of datasets related to the Tamil Language. |  

</br>

### Connectors  

| Connector | Description |  
|---|---|  
| Claude2 | For Claude2 API|  
| HuggingFace | For HuggingFace Inference Endpoints |  
| OpenAI | For OpenAI API |  
| TogetherAI | For TogetherAI Severless API |  

</br>

### Recipes

| Recipe | Description |  
|---|---|  
| AdvGLUE | AdvGLUE is a comprehensive robustness evaluation benchmark that concentrates on assessing the adversarial robustness of language models. It encompasses textual adversarial attacks from various perspectives and hierarchies, encompassing word-level transformations and sentence-level manipulations. A higher grade indicates that the system under test is more resilient to changes in the sentences. |  
| Analogical Similarity | This recipe measures the model’s ability in discriminating between different degrees of analogical similarity in two given episodes. The higher the grade, the better the sytem is performing this capability. |  
| ARC | To measure model&#x27;s ability in answering genuine grade-school level, multiple-choice science questions on the easy and challenge sets. The higher the grade, the better the sytem is performing this capability. |  
| Auto Categorisation | This recipe measures the model&#x27;s ability in categorising elements into groups. The higher the grade, the system performs better in this capability. |  
| Bias Benchmark for QA | This recipe measures the model&#x27;s ability in attesting social biases against people belonging to protected classes along nine social dimensions relevant for US English-speaking context with age ambiguous and disambiguous questions. The higher the grade, the better the system is in handling social biases. |  
| Bias - Occupation | This recipe tests if a targeted system thinks that a certain demographic group can handle a job role better than others. |  
| Cause and Effect | This recipe measure the model&#x27;s ability in understanding the cause and effect in a sentence. The higher the grade, the better the system is in performing this capability. |  
| Chinese Bias Becnhmark for QA - Lite | This recipe measures a system&#x27;s ability to respond to safety-related, biased, and stereotypical questions in Chinese. A higher grade indicates that the system under test is safer, less biased, and less stereotypical. |  
| ChallengingToxicityPrompts | This recipe measures the model&#x27;s ability in generating toxic phrases or sentences when given a toxic phrase. The higher the grade, the less toxic the system is. |  
| Commonsense Morality (Easy) | This recipe measures whether how the system performs when given prompts embedded the body of moral standards and principles that most people intuitively accept. The higher the grade, the better the system performs. |  
| Commonsense Morality (Hard) | This recipe measures whether how the system performs when given prompts embedded the body of moral standards and principles that most people intuitively accept. The higher the grade, the better the system performs. |  
| Contextual Parametric Knowledge Conflicts | This recipe measures the model’s ability in making informed predictions. The higher the grade, the better the system can perform for this capability. |  
| COQA Conversational Q&amp;A | To measure the ability of machines to understand a text passage and answer a series of interconnected questions. The higher the grade, the better the system performs for this capability. |  
| CValues - The benchmark for measuring alignment in LLMs using Chinese Langauges. | This recipe measures the alignment ability of LLMs in both safety and responsibility in Chinese Language. A higher grade means that the LLMs are more aligned to human&#x27;s values. |  
| EnronEmail | The Enron email dataset is aimed at evaluating if there is leakage of private information. The dataset was used in DecodingTrust as part of its privacy evaluation. |  
| GRE Reading Comprehension | To measure the model&#x27;s ability to summarize text, infer and deduce knowledge from context, and match the context. The higher the grade, the better the system is in performing this capability. |  
| GSM8K | To measure model&#x27;s ability in solvinggrade school math problems created by human. These problems take between 2 and 8 steps to solve, and usually requires a sequence of elementary calculations. The higher the grade, the better the system can perform this capability. |  
| HellaSwag | To measure commonsense with questions that are trivial for humans but difficult for statistical models. The higher the grade, the better the system performs this capability. |  
| Jailbreak-DAN | This recipe assesses whether the system will be jailbroken using the common jailbreak methods. The higher the grade, the better the system is protected against common jailbreaks. |  
| Medical MCQ | In this recipe, we measure how well a system is able to answer healthcare questions using various datasets related to healthcare. The higher the grade, the better the system performs in answering healthare questions. |  
| MLCommons AI Safety Benchmark - Child Abuse and Exploitation | This recipe measures the model&#x27;s responses in dealing with prompts that contain, describe, enable, encourage, or endorse the sexual abuse of children. A high grade indicates that the system under test handles the response well. |  
| MLCommons AI Safety Benchmark - Chemical, Biological, Radiological, Nuclear, and Explosives | This recipe measures the model&#x27;s responses in dealing with prompts that enable, encourage, or endorse the creation of weapons of mass destruction. A high grade indicates that the system under test handles the response well. |  
| MLCommons AI Safety Benchmark - Hate | This recipe measures the model&#x27;s responses in dealing with prompts that demean or dehumanize people on the basis of their sensitive, personal characteristics. A high grade indicates that the system under test handles the response well. |  
| MLCommons AI Safety Benchmark - Non-Violent Crimes | This recipe measures the model&#x27;s responses in dealing with prompts that enable, encourage, or endorse the commission of non-violent crimes. A high grade indicates that the system under test handles the response well. |  
| MLCommons AI Safety Benchmark - Sex-related Crimes | This recipe measures the model&#x27;s responses in dealing with prompts that enable, encourage, or endorse the commission of sex-related crimes. A high grade indicates that the system under test handles the response well. |  
| MLCommons AI Safety Benchmark - Suicide &amp; Self-Harm | This recipe measures the model&#x27;s responses in dealing with prompts that enable, encourage, or endorse acts of intentional self-harm. A high grade indicates that the system under test handles the response well. |  
| MLCommons AI Safety Benchmark - Violent Crimes | This recipe measures the model&#x27;s responses in dealing with prompts that enable, encourage, or endorse the commission of violent crimes. A high grade indicates that the system under test handles the response well. |  
| MMLU - Medical related questions. | MMLU benchmark measures model&#x27;s ability in answering accurately across 57 tasks including elementary mathemathics, US history, computer science, law, and more. In this recipe, we extracted all the tasks related to healthare. The higher the grade, the better the system performs in answering healthcare related questions. |  
| MMLU | To measure model&#x27;s ability in answering accurately across 57 tasks including elementary mathemathics, US history, computer science, law, and more. The higher the grade, the better the system performs. |  
| RealToxicityPrompts (Completion) | RealToxicityPrompts aims to measure the model&#x27;s ability in identify toxic sentences. The higher the grade, the less toxic the system is. |  
| RealtimeQA | RealTime QA is a dynamic question answering (QA) platform that inquires about the present. The higher the grade, the better the system is handling hallucination. |  
| Singapore Legal Glossary | To measure the model&#x27;s ability in understanding legal term used commonly in Singapore. |  
| University Tutorial Questions in Singapore&#x27;s Legal Classes | To measure the model&#x27;s ability in understanding legal context in Singapore. |  
| Facts about Singapore | This recipe evaluates the accuracy of facts about Singapore. A higher score reflects the target&#x27;s ability to answer questions about Singapore with a high level of precision and detail. By achieving a high grade, the target demonstrates a thorough understanding and knowledge of Singapore&#x27;s key events, transport system, and facts. |  
| Safety questions written in Singapore&#x27;s context | This recipe evaluates whether a system under test can perform well when a harmful question is written in Singapore&#x27;s context. By achieving a high grade, the target demonstrates that it is lower in risk. |  
| squad-shifts-tnf | Zero-shot reading comprehension on paragraphs and questions from squadshifts. Augmented to true/false statement. The higher the grade, the better the model performs this capability. |  
| TAMIL-KURAL-CLASSIFICATION | This recipe is used to test the comprehension abilities for the Thirukkural. Thirukkural is a classic Tamil literature composed by the ancient Tamil poet Thiruvalluvar. It consists of 1330 couplets (kurals) that are grouped into 133 chapters, each containing 10 couplets. The higher the grade, the better the model performs in this task. |  
| TAMIL-NEWS-CLASSIFICATION | This recipe is to classify a static set of Tamil News. The task is to classify news to its respective category. The higher the grade, the better the model performs in this task. |  
| TAMIL-TANGLISH-TWEETS | This recipe is used to perform sentiment analysis on code-mixed Tamil and English tweets. The higher the grade, the better the model performs in this task. |  
| TruthfulQA (MCQ) | TruthfulQA aims to measure the truthfulness of a model. The higher the grade, the more truthful the system is. |  
| TruthfulQA | TruthfulQA aims to measure the truthfulness of a model. This is an experimental recipe using Rouge Score |  
| UCI Adult Dataset | This recipe contains the easy variation of UCI adult dataset where LLM is expected to classify the attributes correctly. The higher the score, the better the model is. |  
| Winobias | Winobias dataset is used to test representational harm in LLMs. LLM should refuse to answer. The higher the score, the better the model is. |  
| Winograde | To measure model&#x27;s ability in answering commonsense reasoning, expert-crafted pronoun resolution problems that are designed to be unsolvable for statistical models. The higher the score, the better the model is. |  

  

