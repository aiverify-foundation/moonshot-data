import random

import random

import homoglyphs as hg
from moonshot.src.redteaming.attack.attack_module import AttackModule
from moonshot.src.redteaming.attack.attack_module_arguments import AttackModuleArguments
from moonshot.src.utils.log import configure_logger

# Create a logger for this module
logger = configure_logger(__name__)


class RandomHomoglyph(AttackModule):
    def __init__(self, am_id: str, am_arguments: AttackModuleArguments | None = None):
        # Initialize super class
        super().__init__(am_id, am_arguments)
        self.name = "Homoglyph V2 Attack"
        self.description = (
            "This module tests for adversarial textual robustness. Homoglyphs are alternative characters that resemble "
            "a similar ASCII character.\nExample of a homoglyph fool -> fooI"
            "This module slowly increases the percentage"
            "of characters replaced to see how the model reacts to the base prompt."
        )

    def get_metadata(self) -> dict:
        """
        Get metadata for the attack module.

        Returns:
            dict | None: A dictionary containing the 'id', 'name', 'description', 'endpoints' and 'configurations'
            or None if the metadata is not available.
        """
        endpoints = self.req_and_config.get("endpoints", [])
        configurations = self.req_and_config.get("configurations", {})
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description if hasattr(self, "description") else "",
            "endpoints": endpoints,
            "configurations": configurations,
            "configurations": configurations,
        }


    def _get_letter_length(self, word: str) -> int:
        return sum([1 for letter in word if letter.isalnum()] + [0])
        return sum([1 for letter in word if letter.isalnum()] + [0])

    async def execute(self):
        """
        Asynchronously executes the attack module.

        This method loads the dataset contents using the `load_dataset_contents` method,
        processes the dataset through a prompt template, retrieves the connector to the first
        Language Learning Model (LLM) and sends the processed dataset as a prompt to the LLM.
        """
        self.load_modules()
        return await self.perform_attack_manually()

    async def perform_attack_manually(self) -> list:
        """
        Asynchronously performs the attack manually. The user will need to pass in a list of prompts and
        the LLM connector endpoint to send the prompts to. In this example, there is a for loop to send the
        list of prepared prompts to all the LLM connectors defined.

        This method prepares prompts for each target Language Learning Model (LLM) using the provided prompt
        and sends them to the respective LLMs.
        """
        result_list = []
        # Configurble PARAMS - Number of prompts to be sent to target
        # MAX_ITERATION = 20
        # converting glyphs to ASCII characters
        length = self._get_letter_length(self.prompt)

        # process the prompt so we can select only letters from it
        enum_prompt = list(enumerate(self.prompt))
        prompt = list(self.prompt)
        filtered = [item for item in enum_prompt if item[1].isalpha()]

        for i in range(20):
            percentage = i / 20
            percentage = i / 20
            prompt_copy = prompt.copy()
            num_to_replace = int(length * percentage)
            num_to_replace = int(length * percentage)
            letters_to_replace = random.sample(filtered, num_to_replace)
            for index, letter in letters_to_replace:
                try:
                    prompt_copy[index] = random.choice(
                        hg.Homoglyphs().get_combinations(letter)
                    )
                except:
                    logger.error(f"cannot get homoglyph for {letter}")
                    continue
            # homoglyph_prompt = f"[{round(percentage*100)}%]"+"".join(prompt_copy)
            homoglyph_prompt = "".join(prompt_copy)
            result_list.append(await self._send_prompt_to_all_llm([homoglyph_prompt]))


        for res in result_list:
            for x in res:
                logger.debug(
                    f"[HomoglyphGenerator] Prompt: [{round(percentage*100)}% changed] {x.prompt}"
                )
                logger.debug(
                    f"[HomoglyphGenerator] Prompt: [{round(percentage*100)}% changed] {x.prompt}"
                )
                logger.debug(
                    f"[HomoglyphGenerator] Predicted Results: {x.predicted_results}\n"
                )
        return result_list
