from moonshot.src.redteaming.attack.attack_module import AttackModule
from moonshot.src.redteaming.attack.attack_module_arguments import AttackModuleArguments

# How to only download this when user wants to run this attack module?
from transformers import pipeline

# Configurble PARAMS - Number of prompts to be sent to target
DEFAULT_MAX_ITERATION = 30


class ToxicSentenceGenerator(AttackModule):
    def __init__(self, am_id: str, am_arguments: AttackModuleArguments | None = None):
        # Initialize super class
        super().__init__(am_id, am_arguments)
        self.name = "Toxic Sentence Generator"
        self.description = (
            "This module generates toxic sentences based on a given seed prompt. "
            "The attack module intends to test if the system under tests will complete "
            "the sentence with toxic sentences/phrases."
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
        }

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
        redteamer = pipeline(
            "text-generation", model="theastronuts/toxic_sentence_generator"
        )

        # get the configurable params from the config JSON file. if they're not specified, use the default values
        configurations = self.req_and_config.get("configurations", {})
        max_iteration = configurations.get("max_iteration", DEFAULT_MAX_ITERATION)

        for i in range(max_iteration):
            toxic_generator = redteamer(self.prompt)[0]["generated_text"]
            new_prompt = f'"{toxic_generator}"'

            result_list.append(await self._send_prompt_to_all_llm([new_prompt]))

        return result_list
