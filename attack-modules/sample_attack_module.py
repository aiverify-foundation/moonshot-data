from moonshot.src.redteaming.attack.attack_module import AttackModule
from moonshot.src.redteaming.attack.attack_module_arguments import AttackModuleArguments

DEFAULT_SAMPLE_VALUE = "default_value"


class SampleAttackModule(AttackModule):
    def __init__(self, am_id: str, am_arguments: AttackModuleArguments | None = None):
        # Initialize super class
        super().__init__(am_id, am_arguments)
        self.name = "Sample Attack Module"
        self.description = "This is a sample attack module."

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
        configurations = self.req_and_config.get("configurations", {})
        example_field = configurations.get("example_field", DEFAULT_SAMPLE_VALUE)
        print("Example configuration from config JSON file:", example_field)

        result_list = []
        for target_llm_connector in self.connector_instances:
            gen_prompts_generator = self._generate_prompts(
                self.prompt, target_llm_connector.id
            )
            prepared_prompts = [
                prepared_prompt.connector_prompt.prompt
                async for prepared_prompt in gen_prompts_generator
            ]
            result_list.append(
                await self._send_prompt_to_single_llm(
                    prepared_prompts, target_llm_connector
                )
            )
        return result_list
