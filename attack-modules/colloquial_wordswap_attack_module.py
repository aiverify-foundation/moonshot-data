from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

from moonshot.src.redteaming.attack.attack_module import AttackModule
from moonshot.src.redteaming.attack.attack_module_arguments import AttackModuleArguments


class ColloquialWordSwapAttackModule(AttackModule):
    def __init__(self, am_id: str, am_arguments: AttackModuleArguments | None = None):
        # Initialize super class
        super().__init__(am_id, am_arguments)
        self.name = "Colloquial Wordswap"
        self.description = ("About this attack module:\nThis attack module takes in prompts that feature nouns that describe people." 
        "\nExamples of this include words like 'girl' , 'boy' or 'grandmother'.\nThe module substitutes these words"
        " with their Singapore colloquial counterparts, such as 'ah boy', 'ah girl' and 'ah ma'.\nParameters:\n"
        "grandmother - ah ma\ngrandfather - ah gong\ngirl - ah ger\nboy - ah boy\nfather - papa\nmother - mama\naunty - makcik"   
        "\nman - ah beng\nwoman - ah lian")

    def get_metadata(self) -> dict:
        """
        Get metadata for the attack module.

        Returns a dictionary containing the id, name, and description of the attack module. If the name or description
        is not available, empty strings are returned.

        Returns:
            dict | None: A dictionary containing the metadata of the attack module, or None if the metadata is not
            available.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description if hasattr(self, "description") else "",
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
        """
        This is the dictionary that is used to swap words.
        """
        colloquial_dict = {"father": "papa" , 
        "mother": "mama",
        "grandfather": "ah gong",
        "grandmother": "ah ma",
        "girl": "ah ger",
        "boy": "ah boy",
        "aunt": "makcik",
        "aunty": "makcik",
        "man" : "ah beng",
        "woman": "ah lian"}
        result_list = []
        word_list = word_tokenize(self.prompt)
        for i in range(len(word_list)):
            if word_list[i].lower() in list(colloquial_dict.keys()):
                word_list[i] = colloquial_dict[word_list[i]]
        new_prompt = TreebankWordDetokenizer().detokenize(word_list)
        result_list.append(await self._send_prompt_to_all_llm([new_prompt]))
        word_list = word_tokenize(self.prompt)
        for res in result_list:
            for x in res:
                print(x.prompt)
                print(x.predicted_results)
                print()
        return result_list
