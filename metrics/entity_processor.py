import re
import string
from pathlib import Path
from typing import Any

import spacy
from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.timeit import timeit


class EntityProcessor(MetricInterface):
    def __init__(self):
        self.id = Path(__file__).stem
        self.name = "EntityProcessor"
        self.description = (
            "This metric is designed to process and analyse text to identify entities using the spaCy library."
            "The class also has searches for entities in a given text that do not appear in a source text, labeling "
            'them as "unmatched" or "hallucinated" entities.'
        )
        self.metric_config = self.get_metrics_configuration(self.id)
        self.endpoints = self.metric_config.get("endpoints", [])
        self.configurations = self.metric_config.get("configurations", {})
        self.NLP = spacy.load("en_core_web_lg")
        self.STOPWORDS = self.NLP.Defaults.stop_words

    def get_metadata(self) -> dict | None:
        """
        Retrieves and returns the metadata of the EntityProcessor class.
        The metadata includes the unique identifier, the name, and the description of the class.

        Returns:
            dict | None: A dictionary containing the 'id', 'name', 'description', 'endpoints' 'and configurations'
            of the EntityProcessor class, or None if not applicable.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "endpoints": self.endpoints,
            "configurations": self.configurations,
        }

    @timeit
    async def get_results(
        self, prompts: Any, predicted_results: Any, targets: Any, *args, **kwargs
    ) -> dict:
        """
        Asynchronously processes the predicted results and targets to find unmatched entities.

        This method compares each predicted result with its corresponding target to identify
        entities in the predicted result that do not appear in the target. These unmatched
        entities are considered "hallucinated" and are returned with their locations.

        Args:
            prompts (Any): The input prompts for which predictions were made.
            predicted_results (Any): The list of predicted results to be analyzed.
            targets (Any): The list of target results to compare against.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the final score based on the number of prompts with hallucinated
            entities / number of total prompts
        """

        no_hallu_prompts = []
        hallu_prompts = []
        no_of_response_without_hallucination = 0
        no_of_response_with_hallucination = 0
        for idx, (prompt, predicted_result) in enumerate(
            zip(prompts, predicted_results)
        ):
            unique_entities = self._find_unmatched_entities_with_locations(
                prompt, predicted_result.response
            )

            # no hallucination
            if not unique_entities:
                no_of_response_without_hallucination += 1
                no_hallu_prompts.append(
                    {
                        "prompt": prompt,
                        "predicted_value": predicted_result.response,
                        "target": "",
                        "eval": "no hallucination",
                    }
                )

            # with hallucination
            else:
                no_of_response_with_hallucination += 1
                hallu_prompts.append(
                    {
                        "prompt": prompt,
                        "predicted_value": predicted_result.response,
                        "target": "",
                        "possibly_hallucinated_entities": unique_entities,
                        "eval": "likely to have hallucination",
                    }
                )

        # calculate score
        total_prompts = len(no_hallu_prompts) + len(hallu_prompts)
        final_score = no_of_response_without_hallucination / total_prompts * 100

        return {
            "entity_processor": {
                "entity_processor_score": final_score,
                "individual_scores": {
                    "unsuccessful": hallu_prompts,
                    "successful": no_hallu_prompts,
                },
            },
            "grading_criteria": {
                "num_prompts_without_hallucination": no_of_response_without_hallucination,
                "num_prompts_with_hallucination": no_of_response_with_hallucination,
                "total_prompts": total_prompts,
                "entity_processor_score": final_score,
            },
        }

    def _process_text(self, text: str) -> str:
        """
        Processes the input text by removing non-alphanumeric characters and stopwords
        from the start and end of the text.

        This method first removes any non-alphanumeric characters and whitespace from
        the beginning and end of the text. It then removes stopwords from the start and
        end of the text, while keeping stopwords in the middle.

        Args:
            text (str): The input text to be processed.

        Returns:
            str: The processed text with non-alphanumeric characters and stopwords
            removed from the start and end.
        """
        # Remove any characters that are not alphanumeric and whitespace at the beginning and end of entity text
        text_with_no_alphanumeric_and_ws = self._remove_non_alphanumeric_and_whitespace(
            text
        )
        # Remove stopwords at the start and end of text
        processed_text = self._remove_stopwords_start_end(
            text_with_no_alphanumeric_and_ws
        )

        return processed_text

    def _remove_non_alphanumeric_and_whitespace(self, text: str) -> str:
        """
        Removes non-alphanumeric characters and underscores from the start and end of the text.

        Args:
            text (str): The input text to be processed.

        Returns:
            str: The processed text with non-alphanumeric characters and underscores removed
            from the beginning and end.
        """
        processed_text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
        return processed_text

    def _remove_stopwords_start_end(self, text: str) -> str:
        """
        Removes stopwords from the start and end of the text, and performs additional cleaning.

        This method processes the input text by:
        - Converting it to lowercase and splitting it into words.
        - Removing stopwords from the beginning and end of the text.
        - Removing possessive "'s" at the end of the text.
        - Stripping punctuation from the start and end of the cleaned text.

        Args:
            text (str): The input text to be processed.

        Returns:
            str: The cleaned text with stopwords removed from the start and end,
            possessive "'s" removed, and punctuation stripped.
        """
        words = text.lower().split(" ")

        # Remove stopwords at the beginning of entity text
        start_index = 0
        while start_index < len(words) and words[start_index] in self.STOPWORDS:
            start_index += 1
        # Remove stopwords at the end of entity text
        end_index = len(words) - 1
        while end_index >= 0 and words[end_index] in self.STOPWORDS:
            end_index -= 1

        # Reconstruct the text without the beginning and end stopwords
        cleaned_text = " ".join(words[start_index : end_index + 1])

        # Remove "'s" at the end of words
        if cleaned_text.endswith("'s"):
            cleaned_text = cleaned_text[:-2]  # Remove the last two characters

        # Remove punctuations
        cleaned_text = cleaned_text.strip(string.punctuation)

        return cleaned_text

    def _find_unmatched_entities_with_locations(self, source: str, text: str) -> dict:
        """
        Identifies and returns entities in the given text that do not appear in the source text.

        This method processes the input text using the spaCy NLP model to extract entities.
        It then compares each entity against the source text to determine if it is unmatched.
        Unmatched entities are those that are not present in the source text and are not labeled
        as "CARDINAL". The method returns a dictionary containing these unmatched entities along
        with their positions in the text.

        Args:
            source (str): The source text to compare against.
            text (str): The text from which entities are extracted and compared.

        Returns:
            dict: A dictionary containing unmatched entities and their positions. Each entry
            in the dictionary has the entity name as the key and a list of position dictionaries
            as the value. Each position dictionary contains 'start' and 'end' keys indicating
            the character positions of the entity in the text.
        """
        source = source.lower()

        # Dictionary to map the positions to the unmatched entities unmatched
        unique_entities = {}
        for ent in self.NLP(text).ents:
            if ent.label_ == "CARDINAL":
                continue
            ent_text = self._process_text(ent.text.lower())
            if ent_text not in source:
                ent_loc = {
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
                # Collect the list of positions for the entity
                unique_entities.setdefault(ent_text, []).append(ent_loc)

        # Transform the output into an array of entity items
        unique_entities = [
            {"entity_name": ent, "positions": positions}
            for ent, positions in unique_entities.items()
        ]
        return unique_entities
