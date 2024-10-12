import re
import time
from pathlib import Path
from typing import Any

import spacy
from moonshot.src.connectors.connector import Connector
from moonshot.src.connectors_endpoints.connector_endpoint import ConnectorEndpoint
from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.log import configure_logger
from moonshot.src.utils.timeit import timeit
from sentence_transformers import SentenceTransformer

# Create a logger for this module
logger = configure_logger(__name__)

DEFAULT_EVALUATION_MODEL = "openai-gpt4"


class FactScore(MetricInterface):
    """
    FactScore returns the various fact scores.
    """

    break_prompt = """Split the following document (delimited by ```) to sentences:
    Document: ```{document}```
    Output the sentences in JSON format with the following key: 'list_of_sentences'.
    """

    fact_check_prompt = """You are a careful fact-checker. Below is a reference document (delimited by ```):
    Document: ```{document}```
    According to the above reference document, is the following hypothesis: \
    true (entailment), false (contradiction), or undetermined (neutral)?
    Hypothesis: {statement}
    If the hypothesis is not true, revise it to be consistent with the reference document.
    Provide output in JSON format with the following four keys:
    'hypothesis', 'decision': (true, false, or undetermined), 'reason', 'revision'.
    """

    def __init__(self):
        """
        Initialize the FactScore class with default values and configurations.
        """
        self.id = Path(__file__).stem
        self.name = "Factscore"
        self.description = (
            "Factscore is a metric designed to evaluate the factual accuracy of statements "
            "by comparing them against a reference document."
        )
        self.metric_config = self.get_metrics_configuration(self.id)
        self.endpoints = self.metric_config.get("endpoints", [DEFAULT_EVALUATION_MODEL])
        self.configurations = self.metric_config.get("configurations", {})

        # Factscore specific attributes
        self.evaluation_model = [
            Connector.create(ConnectorEndpoint.read(ep_id))
            for ep_id in self.metric_config["endpoints"]
        ][0]
        self.extract_facts_with_local_model = self.configurations.get(
            "extract_facts_with_local_model", True
        )
        self.length_limit = self.configurations.get("length_limit", 10000)
        # self.perform_postparsing = self.configurations.get("perform_postparsing", False)

        # Factscore variables
        self.pattern = re.compile(
            r": \(?(at )?\[\d*(\(\w\)(\(i{1,3}\))?)?].*?\.", re.DOTALL
        )

        # Load models
        self.load_models()

    def load_models(self):
        """
        Load the necessary models for FactScore.
        """
        logger.info("[FactScore] Loading spacy model 'en_core_web_trf'.")
        start_time = time.perf_counter()
        self.nlp = spacy.load("en_core_web_trf")
        logger.info(
            f"[FactScore] Loading spacy model 'en_core_web_trf' took {(time.perf_counter() - start_time):.4f}s"
        )

        logger.info(
            "[FactScore] Loading sentence transformer 'sentence-transformers/all-mpnet-base-v2'."
        )
        start_time = time.perf_counter()
        self.sbert_model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2"
        )
        logger.info(
            f"[FactScore] Loading sentence transformer 'sentence-transformers/all-mpnet-base-v2' took "
            f"{(time.perf_counter() - start_time):.4f}s"
        )

    @timeit
    def get_metadata(self) -> dict | None:
        """
        Retrieves and returns the metadata of the FactScore class.

        Returns:
            dict | None: A dictionary containing the 'id', 'name', 'description', 'endpoints' and 'configurations' of
            the FactScore class, or None if not applicable.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "endpoints": self.endpoints,
            "configurations": self.configurations,
        }

    def extract_facts(self, input_document: str, prompt: str = break_prompt) -> list:
        """
        Split text document to sentences using ChatGPT

        Args:
            input_document: input text document
            prompt: prompt for text splitting

        Return:
            a list of dict [{'list_of_sentences'}]
            length of list: number of paragraphs in the input document
        """
        # Split document to paragraphs
        paragraphs = input_document.split("\n")

        if self.extract_facts_with_local_model:
            logger.info("[FactScore] Extracting facts using local model.")
            facts_dicts = [
                {
                    "list_of_sentences": self.split_paragraph_to_sentences(
                        paragraph, cleaning=False
                    )
                }
                for paragraph in paragraphs
                if paragraph
            ]
            return facts_dicts
        else:
            logger.info("[FactScore] Extracting facts using external model.")
            prompts_list = [
                prompt.format(document=paragraph)
                for paragraph in paragraphs
                if paragraph
            ]
            return self.batch_call_api(prompts_list)

    def check_facts(self, facts_list: list, prompt: str = fact_check_prompt) -> list:
        """
        Check the fact in the list against reference

        Args:
            facts_list: a list of dict
            prompt: prompt for fact-checking

        Return:
            a list of dict: [{'hypothesis', 'decision', 'reason', 'revision'}]
        """
        if not facts_list:
            return []
        prompts_list = [
            prompt.format(document=tmp_fact["reference"], statement=tmp_fact["fact"])
            for tmp_fact in facts_list
        ]
        return self.batch_call_api(prompts_list)

    def batch_call_api(self, prompts_list: list) -> list:
        """
        Call the API in batches to get predictions.

        Args:
            prompts_list: List of prompts to be sent to the API.

        Returns:
            List of predicted results.
        """
        # Format the targets and output response
        prompts_info = {
            "data": [
                {
                    "prompt": prompt,
                }
                for prompt in prompts_list
            ]
        }

        # Compute factscore for all input samples
        extracted_facts = Connector.get_predictions(
            prompts_info,
            self.conn_instance,
            None,
        )

        # Combine the extracted facts
        return [tmp_fact["predicted_result"] for tmp_fact in extracted_facts]

    def check_facts_in_completions(
        self, facts_completions: list[dict[str, list[str]]], ref_dict: dict[str, list]
    ) -> tuple:
        """
        This function checks the facts in the completions against the reference document.

        Args:
            facts_completions: sentences from summary
            ref_dict: reference

        Return:
            fact_check_results: a list of dict: [{'fact', 'max_score', 'decision', 'reason', 'revision'}]
            total_facts: an integer for the total number of facts checked
            total_bad_facts: an integer for the total number of bad facts found
        """
        fact_check_results = []

        # total sentences, total_facts = checked_facts + num of short sentence + Num of sentences with max_score >= 0.9
        total_facts = 0

        # total num of sentences sent to GPT4
        checked_facts = 0

        # total num of sentences labelled as bad by OpenAI model
        total_bad_facts = 0

        # GPT4 cannot determine
        undetermined = 0

        # No. of sentences with error
        error = 0

        for completion in facts_completions:
            # Retrieve the reference for each completion
            retrieval_output = self.retrieve_reference(
                completion["list_of_sentences"], ref_dict, self.length_limit
            )

            # Create a list of facts to check
            fact_check_list = [
                {"index": i, "fact": d["sentence"], "reference": d["reference"]}
                for i, d in enumerate(retrieval_output)
                if d.get("max_score") is not None and d["max_score"] < 0.9
                # Skip facts with very high similarity scores
            ]

            # prepare the list of sentences to be sent to GPT4
            fact_check_list = list()
            for idx, sent in enumerate(retrieval_output):
                if sent.get("max_score", None) is None:
                    continue
                if sent["max_score"] >= 0.9:
                    continue
                else:
                    fact_check_list.append(
                        {
                            "index": idx,
                            "fact": sent["sentence"],
                            "reference": sent["reference"],
                        }
                    )
                    checked_facts += 1

            fact_check_completions = self.check_facts(fact_check_list)

            # Prepare the output data
            fact_check_output = self.prepare_output_data(
                retrieval_output, fact_check_completions, fact_check_list
            )
            fact_check_results.append(fact_check_output)

            # Compute total facts and total bad facts
            total_facts += sum(1 for d in fact_check_output if d["max_score"])
            total_bad_facts += sum(
                1
                for d in fact_check_output
                if d["max_score"] and str(d.get("decision", "true")).lower() == "false"
            )
            error += sum(
                1
                for d in fact_check_output
                if d["max_score"] and str(d.get("decision", "true")).lower() == "error"
            )
            undetermined += sum(
                1
                for d in fact_check_output
                if d["max_score"]
                and str(d.get("decision", "true")).lower() == "undetermined"
            )

        return (
            fact_check_results,
            total_facts,
            checked_facts,
            total_bad_facts,
            error,
            undetermined,
        )

    def prepare_output_data(
        self, retrieval_output: list, fact_check_completions: list, facts_list: list
    ) -> list:
        """
        Prepare the output data for fact checking.

        Args:
            retrieval_output (list): A list of dictionaries representing the retrieval output.
            fact_check_completions (list): A list of dictionaries representing the fact check completions.
            facts_list (list): A list of dictionaries representing the facts list.

        Returns:
            list: The prepared output data for fact checking.
        """
        fact_check_output = [
            {"fact": d["sentence"], "max_score": d.get("max_score", "")}
            for d in retrieval_output
        ]
        for i, d in enumerate(fact_check_completions):
            if isinstance(d, dict) and "hypothesis" in d:
                d.pop("hypothesis")
                fact_check_output[facts_list[i]["index"]].update(d)
            else:
                logger.error(
                    f"[FactScore] An error trying to update fact check. Setting decision as an error.\n"
                    f"Index: ({facts_list[i]['index']}) Type: {type(d)} Contents: {d}"
                )
                fact_check_output[facts_list[i]["index"]].update({"decision": "error"})
        return fact_check_output

    def compute_factscore_helper(
        self, reference: str, candidate: str
    ) -> tuple[bool, dict]:
        """
        Compute FactScore for evaluating the factual consistency of generated summary.

        Args:
            reference: the source document
            candidate: the generated summary

        Returns:
            dict: {'factscore': {'total_facts', 'total_bad_facts', 'factscore', 'run_time', 'revision', 'results'}}
            'results': a list of dict: [{'fact', 'max_score', 'decision', 'reason', 'revision'}]
        """
        try:
            total_start_time = time.perf_counter()

            # Split the reference document into sentences
            logger.info("[FactScore] Splitting document to sentences")
            start_time = time.perf_counter()
            reference_dict = self.split_document_to_sentences(reference)
            doc_to_str_duration = f"{(time.perf_counter() - start_time):.4f}s"

            # Extract facts from the candidate document
            logger.info("[FactScore] Extracting facts from candidate document")
            start_time = time.perf_counter()
            facts_completions = self.extract_facts(candidate)
            extraction_facts_duration = f"{(time.perf_counter() - start_time):.4f}s"

            # Check the extracted facts against the reference document
            logger.info("[FactScore] Checking facts against reference document")
            start_time = time.perf_counter()
            (
                fact_check_results,
                total_facts,
                checked_facts,
                total_bad_facts,
                error,
                undetermined,
            ) = self.check_facts_in_completions(facts_completions, reference_dict)
            check_facts_duration = f"{(time.perf_counter() - start_time):.4f}s"

            # Compute the fact score
            factscore = (
                1 - total_bad_facts / (total_facts - error - undetermined)
                if total_facts
                else ""
            )

            # Compute run time
            run_duration = f"{(time.perf_counter() - total_start_time):.4f}s"

            return True, {
                "reference": reference,
                "candidate": candidate,
                "total_facts": total_facts,
                "checked_facts": checked_facts,
                "bad_facts": total_bad_facts,
                "error": error,
                "undetermined": undetermined,
                "factscore": factscore,
                "doc_to_str_duration": doc_to_str_duration,
                "extraction_facts_duration": extraction_facts_duration,
                "check_facts_duration": check_facts_duration,
                "total_run_duration": run_duration,
                "results": fact_check_results,
            }
        except ConnectionError as conn_error:
            logger.error(f"[FactScore] Failed to compute factscore: {str(conn_error)}")
            raise conn_error
        except Exception as error:
            logger.warning(f"[FactScore] Failed to compute factscore: {str(error)}")
            return False, {
                "reference": reference,
                "candidate": candidate,
            }

    @timeit
    def compute_factscore(self) -> dict:
        """
        Compute factscores for an input list of source document and summary pairs

        Returns:
            a dict: {'factscore', 'results'}
        """
        start_perf_time = time.perf_counter()

        # Load model endpoint and set db instance
        self.conn_instance = Connector.load_from_json_config(self.model_endpoint)

        # individual_factscore contains the statistics from each document in a task
        individual_factscore = []
        for index, (reference, candidate) in enumerate(
            zip(self.targets, self.output_response)
        ):
            # # Check if you need to message the candidate
            # if self.perform_postparsing:
            #     # Perform prompt processing
            #     if isinstance(reference, list):
            #         reference = self.slr_extract_judgment(reference)
            #     else:
            #         reference = self.slr_extract_judgment(json.loads(reference))

            #     # Perform candidate processing
            #     candidate = "Facts\n\n" + candidate.split("Facts\n\n")[1]

            is_success, result = self.compute_factscore_helper(reference, candidate)
            individual_factscore.append(result)
            if is_success:
                logger.info(f"[FactScore] #{index} computed")
            else:
                logger.warning(f"[FactScore] #{index} error")

        # the average statistics w.r.t a task
        factscore_stats = {
            "total_facts": 0,
            "total_checked_facts": 0,
            "total_bad_facts": 0,
            "total_error": 0,
            "total_undetermined": 0,
            "factscore": 0.0,
            "total_runtime": "",
            "llm-endpoint": "",
        }

        for result in individual_factscore:
            if result:
                factscore_stats["total_facts"] += result.get("total_facts", 0)
                factscore_stats["total_checked_facts"] += result.get("checked_facts", 0)
                factscore_stats["total_bad_facts"] += result.get("bad_facts", 0)
                factscore_stats["total_error"] += result.get("error", 0)
                factscore_stats["total_undetermined"] += result.get("undetermined", 0)

        if factscore_stats["total_facts"] > 0:
            factscore_stats["factscore"] = (
                1 - factscore_stats["total_bad_facts"] / factscore_stats["total_facts"]
            )

        # Compute run time
        end_time = time.perf_counter()
        run_duration = f"{(end_time - start_perf_time):.4f}s"

        factscore_stats["total_runtime"] = run_duration
        factscore_stats["llm-endpoint"] = self.model_endpoint

        return {
            "individual_scores": individual_factscore,
            "average_scores": factscore_stats,
        }

    @timeit
    async def get_results(
        self, prompts: Any, predicted_results: Any, targets: Any, *args, **kwargs
    ) -> dict:
        """
        Calculate and return the results.

        Parameters:
            prompts (Any): The prompts used for generating the predicted results.
            predicted_results (Any): The predicted results generated by the model.
            targets (Any): The target results for comparison.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            dict: A dictionary containing my results
        """
        logger.info("[FactScore] Getting results...")

        # Compute factscore
        output_results = self.compute_factscore()

        # Return the final rouge scores dictionary
        output_results.update({"grading_criteria": {}})
        return output_results

    def remove_ref(self, s: str) -> str:
        """
        To replace the reference of the following format to period ".":
            ": at [12], [34] and [56]."
            ": [12], [34] and [56]."
            ": at [40(a)], [74] to [79] and [83(e)]."
            ": (at [64] and [68])."
            ": at [25(c)(ii)]."
            Parameters:
                s (str): input text
            Returns:
                text (str): output of all references matching the formats removed.
        """
        return self.pattern.sub(".", s)

    def remove_list_number(self, text: str) -> str:
        """
        Remove list number at the beginning of a text paragraph.

        Args:
            text (str): The input text paragraph.

        Returns:
            str: The text paragraph with the list number removed.
        """
        words = text.split(maxsplit=1)
        if words[0].strip("().").isdigit():
            return words[1] if len(words) > 1 else ""
        return text

    def clean_text(self, text: str) -> str:
        """
        Clean the text by removing paragraph numbers at the beginning and references at the end.

        Args:
            text (str): The input text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        return " ".join(self.remove_ref(self.remove_list_number(text)).split())

    def fit_paragraphs_to_limit(self, word_counts: list, length_limit: int) -> int:
        """
        Check how many paragraphs can fit into the length limit.

        Args:
            word_counts (list): Number of words in each paragraph.
            length_limit (int): The total number of words in selected paragraphs should not exceed this limit.

        Returns:
            int: Number of paragraphs that can fit into the length limit.
        """
        total_words = 0
        for i, count in enumerate(word_counts):
            total_words += count
            if total_words > length_limit:
                return i
        return len(word_counts)

    def find_top_reference(self, scores, ref_dict: dict, length_limit: int) -> dict:
        """
        Find indexes of the most relevant paragraphs in reference.

        Args:
            scores (numpy array): A numpy array of similarity scores.
            length_limit (int): The total number of words in selected paragraphs should not exceed this limit.
            ref_dict (dict): A dictionary of reference containing 'parags', 'word_counts', 'sents', 'idx'.

        Returns:
            dict: A dictionary containing 'max_score' and 'idx' (indexes of the most relevant reference paragraphs).
        """
        # Sort reference sentences on scores in descending order
        scores_dict = sorted(
            [
                {
                    "i": parag_index,
                    "score": score,
                    "word_count": ref_dict["word_counts"][parag_index],
                }
                for parag_index, score in zip(ref_dict["idx"], scores)
            ],
            key=lambda x: x["score"],
            reverse=True,
        )

        # Remove duplicates and keep only the first occurrence
        seen = set()
        scores_dict = [
            d for d in scores_dict if not (d["i"] in seen or seen.add(d["i"]))
        ]

        # Find how many reference paragraphs can fit into length limit
        number_of_parags = self.fit_paragraphs_to_limit(
            word_counts=[d["word_count"] for d in scores_dict],
            length_limit=length_limit,
        )

        return {
            "max_score": round(float(scores_dict[0]["score"]), 4),
            "idx": sorted([d["i"] for d in scores_dict[:number_of_parags]]),
        }

    def retrieve_reference(
        self, sentences: list, ref_dict: dict, length_limit: int
    ) -> list:
        """
        Retrieve relevant reference paragraphs for a list of candidate sentences.

        Args:
            sentences (list): A list of candidate sentences.
            ref_dict (dict): A dictionary of reference containing 'parags', 'word_counts', 'sents', 'idx'.
            length_limit (int): The total number of words in selected paragraphs should not exceed this limit.

        Returns:
            list: A list of dictionaries containing 'sentence', 'cleaned', 'reference', and 'max_score'.
        """
        # Select candidate sentences (>= 3 words) for similarity computation
        candidates = [
            {"index": i, "sentence": sent, "cleaned": self.clean_text(sent)}
            for i, sent in enumerate(sentences)
        ]

        selected_candidates = [
            cand for cand in candidates if len(cand["cleaned"].split()) >= 3
        ]

        if not selected_candidates:
            return candidates

        # Compute similarity scores
        sim_scores = self.compute_sbert_scores(
            ref_dict["sents"], [cand["cleaned"] for cand in selected_candidates]
        )
        logger.info(f"[FactScore] Similarity scores (shape): {sim_scores.shape}")

        # Retrieve top reference paragraphs for each selected sentence
        for i, cand in enumerate(selected_candidates):
            d = self.find_top_reference(
                scores=sim_scores[:, i], ref_dict=ref_dict, length_limit=length_limit
            )

            candidates[cand["index"]]["max_score"] = d["max_score"]
            candidates[cand["index"]]["reference"] = "\n".join(
                ref_dict["parags"][index] for index in d["idx"]
            )

        return candidates

    def split_paragraph_to_sentences(self, text: str, cleaning=True) -> list:
        """
        Split a text paragraph into a list of sentences, removing short sentences with less than 3 words.

        Args:
            text (str): The input text paragraph.
            cleaning (bool): Whether to clean the text before splitting.

        Returns:
            list: A list of sentences.
        """
        if cleaning:
            text = self.clean_text(re.sub(r";", ".", text))
            return [
                " ".join(sent.text.split())
                for sent in self.nlp(text).sents
                if len(sent.text.split()) >= 3
            ]
        else:
            return [sent.text for sent in self.nlp(text).sents]

    def split_document_to_sentences(self, text: str) -> dict:
        """
        Split a document first into paragraphs and then into sentences.

        Args:
            text (str): The input document text.

        Returns:
            dict: A dictionary containing 'parags', 'word_counts', 'sents', and 'idx'.
        """

        # Split document to paragraphs
        def exclude_empty_str(x: str) -> bool:
            # remove empty string case
            if x is None:
                return False
            if not x.strip():
                return False
            return True

        parags = list(filter(exclude_empty_str, text.split("\n")))
        word_counts = [len(parag.split()) for parag in parags]

        # Split paragraph to sentences
        sents = []
        idx = []
        for i, parag in enumerate(parags):
            strs = self.split_paragraph_to_sentences(parag)
            sents.extend(strs)
            idx.extend([i] * len(strs))

        return {
            "parags": parags,
            "word_counts": word_counts,
            "sents": sents,
            "idx": idx,
        }

    def compute_sbert_scores(self, ref_strs: list, test_strs: list):
        """
        Use Sentence Transformer to compute cosine similarity scores between reference strings and test strings.

        Args:
            ref_strs (list): A list of reference strings.
            test_strs (list): A list of test strings.

        Returns:
            numpy array: Cosine similarity scores in an M x N numpy array, where M is the length of ref_strs and
                         N is the length of test_strs.
        """
        start = time.time()
        embeddings1 = self.sbert_model.encode(ref_strs)
        embeddings2 = self.sbert_model.encode(test_strs)
        scores = self.np.inner(embeddings1, embeddings2)
        end = time.time()
        logger.info(
            f"[FactScore] Time for similarity computation (SBert): {(end - start):.1f} seconds"
        )
        return scores

    # def slr_extract_judgment(self, jsondict: list) -> str:
    #     """
    #     Extract Judgment text from SAL JSON dict data.

    #     Args:
    #         jsondict (list): A list of dictionaries representing the SAL JSON data.

    #     Returns:
    #         str: The extracted judgment text.
    #     """
    #     output_buf = []
    #     for data in jsondict:
    #         # Extract header text
    #         output_buf.append(data["header"]["text"])

    #         # Extract paragraph text
    #         for parag_data in data["paragraphs"]:
    #             if parag_data["paragraph_number"]:
    #                 output_buf.append(
    #                     parag_data["paragraph_number"] + " " + parag_data["text"]
    #                 )
    #             else:
    #                 output_buf.append(parag_data["text"])

    #             # Extract table text
    #             for table_data in parag_data["tables"]:
    #                 rows = [row.replace("\t", " | ") for row in table_data]
    #                 output_buf.append("\n".join(rows))

    #     text = "\n\n".join(output_buf)
    #     return text
