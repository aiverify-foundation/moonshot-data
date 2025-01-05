from __future__ import annotations

import ast
import asyncio
import json
import random
from datetime import datetime
from itertools import groupby
from operator import attrgetter
from typing import AsyncGenerator, Callable

from jinja2 import Template
from moonshot.src.configs.env_variables import EnvVariables
from moonshot.src.connectors.connector import Connector
from moonshot.src.connectors.connector_prompt_arguments import ConnectorPromptArguments
from moonshot.src.connectors.connector_response import ConnectorResponse
from moonshot.src.connectors_endpoints.connector_endpoint import ConnectorEndpoint
from moonshot.src.cookbooks.cookbook import Cookbook
from moonshot.src.datasets.dataset import Dataset
from moonshot.src.metrics.metric import Metric
from moonshot.src.recipes.recipe import Recipe
from moonshot.src.results.result_arguments import ResultArguments
from moonshot.src.runs.run_progress import RunProgress
from moonshot.src.runs.run_status import RunStatus
from moonshot.src.storage.db_interface import DBInterface
from moonshot.src.storage.storage import Storage
from moonshot.src.utils.atomic_integer import AtomicInteger
from moonshot.src.utils.log import configure_logger
from pydantic import BaseModel

# Create a logger for this module
logger = configure_logger(__name__)

# ---------------------------------------------------------------------------------------------------------------------
# BenchmarkingTask Messages
# ---------------------------------------------------------------------------------------------------------------------
BENCHMARKINGTASK_EXECUTE_METRICS_CALCULATION_MSG = "[BenchmarkingTask] Executing metric '{metric_name}' with {num_of_completed_prompts} completed prompts."  # noqa: E501
BENCHMARKINGTASK_EXECUTE_LLM_QUERY_ERROR_MSG = "[BenchmarkingTask] Error: Prompts generator is not initialized. Set it up before executing the LLM query."  # noqa: E501
BENCHMARKINGTASK_SETUP_ERROR = "[BenchmarkingTask] Error during setup: {message}"
BENCHMARKINGTASK_GET_DATASET_PROMPTS_MSG = "[BenchmarkingTask] Using {num_of_prompts} out of {total_dataset_prompts} prompts from dataset {ds_id}."  # noqa: E501
# ---------------------------------------------------------------------------------------------------------------------
# Benchmarking Messages
# ---------------------------------------------------------------------------------------------------------------------
BENCHMARKING_CANCEL_SET_WARNING = (
    "[Benchmarking] Cancellation flag detected. Halting process..."
)
BENCHMARKING_GENERATE_ERROR = (
    "[Benchmarking] Error encountered during result generation: {message}"
)
BENCHMARKING_GENERATE_NO_DATABASE_INSTANCE_MSG = (
    "Database instance missing. Terminating process."
)
BENCHMARKING_GENERATE_TASKS_COOKBOOK_ERROR_MSG = (
    "Error loading cookbook '{cookbook_name}': {message}. Terminating process."
)
BENCHMARKING_GENERATE_TASKS_RECIPE_ERROR_MSG = (
    "Error loading recipe '{recipe_name}': {message}. Terminating process."
)
BENCHMARKING_GENERATE_UPDATE_BENCHMARK_STATUS = (
    "[Benchmarking] Benchmark status update in progress."
)
BENCHMARKING_GET_CONNECTOR_ENDPOINT_INSTANCES_LOAD_FAILED_MSG = (
    "Error loading endpoint '{endpoint_name}': {message}. Terminating process."
)
BENCHMARKING_RUN_CHECK_TASKS_PROMPTS_METRICS = "[Benchmarking] No benchmark tasks, prompts, or metrics available. Exiting the process."  # noqa: E501
BENCHMARKING_RUN_DEBUG_MSG = (
    "[Benchmarking] Executing with parameters: run_id={run_id}, runner_id={runner_id}, runner_args={runner_args}, "
    "database_instance={database_instance}, endpoints={endpoints}, run_progress={run_progress}, cancel_event={cancel_event}"  # noqa: E501
)
BENCHMARKING_RUN_GENERATE_TASKS_INFO = "[Benchmarking] Created {message} tasks with a total of {total_prompts} prompts and {total_metrics} metrics."  # noqa: E501
BENCHMARKING_RUN_LOAD_ENDPOINT_INSTANCES_NO_ENDPOINTS_MSG = (
    "No connector endpoints specified. Exiting."
)
BENCHMARKING_RUN_LOAD_ENDPOINT_INSTANCES_SUCCESS = (
    "[Benchmarking] Successfully loaded connector endpoints."
)
BENCHMARKING_RUN_QUERY_LLM_INFO = (
    "[Benchmarking] Initiating LLM queries for {num_of_tasks} task(s)."
)
BENCHMARKING_RUN_UPDATE_SYSTEM_PROMPT_SKIPPED = (
    "[Benchmarking] Connector system prompts update skipped."
)
BENCHMARKING_RUN_UPDATE_SYSTEM_PROMPT_SUCCESS = (
    "[Benchmarking] Connector system prompts updated successfully."
)
BENCHMARKING_SET_SYSTEM_PROMPTS_FAILED_MSG = (
    "Unable to set system prompt on '{connector_name}': {message}. Exiting."
)


# ---------------------------------------------------------------------------------------------------------------------
# Benchmarking Prompt Class
# ---------------------------------------------------------------------------------------------------------------------
class BenchmarkingPrompt(BaseModel):
    """
    Represents a benchmarking prompt with necessary identifiers and prompt arguments.

    Attributes:
        connection_id (str): The ID of the connection. Defaults to an empty string.
        recipe_id (str): The ID of the recipe associated with the prompt.
        dataset_id (str): The ID of the dataset used in the benchmarking.
        prompt_template_id (str): The ID of the prompt template.
        connector_prompt (ConnectorPromptArguments): The prompt information to be sent to the connector.
    """

    connection_id: str = ""  # The ID of the connection, default is an empty string
    recipe_id: str  # The ID of the recipe
    dataset_id: str  # The ID of the dataset
    prompt_template_id: str  # The ID of the prompt template
    connector_prompt: ConnectorPromptArguments  # The prompt information to send

    def to_tuple(self) -> tuple:
        """
        Convert the BenchmarkingPrompt instance into a tuple for serialization.

        This method compiles the attributes of the BenchmarkingPrompt instance into a tuple.
        The tuple includes the following attributes in order:
        connection_id, recipe_id, dataset_id, prompt_template_id, prompt_index, prompt, target,
        predicted_results, and duration.

        This tuple format is useful for serialization, such as storing the BenchmarkingPrompt data
        in a database or transmitting it across network boundaries.

        Returns:
            tuple: A tuple containing the serialized attributes of the BenchmarkingPrompt instance.
        """
        return (
            self.connection_id,
            self.recipe_id,
            self.dataset_id,
            self.prompt_template_id,
            self.connector_prompt.prompt_index,
            self.connector_prompt.prompt,
            str(self.connector_prompt.target),
            json.dumps(self.connector_prompt.predicted_results.to_dict()),
            str(self.connector_prompt.duration),
        )

    @classmethod
    def from_tuple(cls, cache_record: tuple) -> BenchmarkingPrompt:
        """
        Reconstitutes a BenchmarkingPrompt instance from a tuple representation.

        This method accepts a tuple with values that map to the attributes of a BenchmarkingPrompt object.
        The expected order of values in the tuple is:
        connection_id, recipe_id, dataset_id, prompt_template_id, prompt_index, prompt, target,
        predicted_results, and duration. It constructs a new BenchmarkingPrompt instance using these values.
        The primary purpose of this method is to recreate BenchmarkingPrompt instances from their serialized form,
        such as data retrieved from a database or received over a network.

        Args:
            cache_record (tuple): A tuple with ordered values that map to the properties of a
            BenchmarkingPrompt instance.

        Returns:
            BenchmarkingPrompt: An instance of BenchmarkingPrompt initialized with the data from the tuple.
        """
        # The target and predicted_results fields may be stored as strings in the cache_record.
        # ast.literal_eval is used to attempt to convert these strings back into their original data types.
        # If the conversion fails (i.e., the fields are not string representations of Python literals),
        # the original string values are used.
        try:
            target = ast.literal_eval(cache_record[9])
        except Exception:
            target = cache_record[9]

        try:
            predicted_results_dict = json.loads(cache_record[10])
            predicted_results = ConnectorResponse(**predicted_results_dict)
        except Exception:
            predicted_results = cache_record[10]

        return cls(
            connection_id=cache_record[1],
            recipe_id=cache_record[2],
            dataset_id=cache_record[3],
            prompt_template_id=cache_record[4],
            connector_prompt=ConnectorPromptArguments(
                prompt_index=cache_record[7],
                prompt=cache_record[8],
                target=target,
                predicted_results=predicted_results,
                duration=float(cache_record[11]),
            ),
        )


# ---------------------------------------------------------------------------------------------------------------------
# Benchmarking Task Class
# ---------------------------------------------------------------------------------------------------------------------
class BenchmarkingTask:
    """
    Represents the configuration and state for executing a benchmarking task.

    Attributes:
        recipe_name (str): The name of the recipe used in the benchmarking task.
        recipe_instance (Optional[Recipe]): An instance of the Recipe class, if available.
        connector (Connector): The connector employed for the benchmarking task.
        prompt_selection_percentage (int): The percentage of prompts selected for the task.
        random_seed (int): The seed value for random operations to ensure reproducibility.
        use_cache (bool): Indicates whether to use cached results.
        start_time (datetime): The timestamp marking the start of the benchmarking task.
        end_time (datetime): The timestamp marking the end of the benchmarking task.
        num_of_total_metrics (AtomicInteger): The total count of metrics to be processed.
        num_of_completed_metrics (AtomicInteger): The count of metrics that have been completed.
        num_of_error_metrics (AtomicInteger): The count of metrics that resulted in errors.
        num_of_cancelled_metrics (AtomicInteger): The count of metrics that were cancelled.
        num_of_total_prompts (AtomicInteger): The total count of prompts to be processed.
        num_of_completed_prompts (AtomicInteger): The count of prompts that have been completed.
        num_of_error_prompts (AtomicInteger): The count of prompts that resulted in errors.
        num_of_cancelled_prompts (AtomicInteger): The count of prompts that were cancelled.
        completed_benchmark_prompts (List[BenchmarkingPrompt]): A list of successfully completed benchmarking prompts.
        error_benchmark_prompts (List[BenchmarkingPrompt]): A list of benchmarking prompts that resulted in errors.
        cancelled_benchmark_prompts (List[BenchmarkingPrompt]): A list of benchmarking prompts that were cancelled.
        completed_benchmark_metrics (List[str]): A list of successfully completed benchmarking metrics.
        error_benchmark_metrics (List[str]): A list of benchmarking metrics that resulted in errors.
        cancelled_benchmark_metrics (List[str]): A list of benchmarking metrics that were cancelled.
    """

    sql_create_runner_cache_record = """
        INSERT INTO runner_cache_table(connection_id,recipe_id,dataset_id,prompt_template_id,
        prompt_index,prompt,target,predicted_results,duration)
        VALUES(?,?,?,?,?,?,?,?,?)
    """
    sql_read_runner_cache_record = """
        SELECT * from runner_cache_table WHERE connection_id=? AND recipe_id=?
        AND dataset_id=? AND prompt_template_id=? AND prompt=?
    """

    def __init__(
        self,
        database_instance: DBInterface,
        recipe_name: str,
        recipe_instance: Recipe | None,
        connector: Connector,
        prompt_selection_percentage: int,
        random_seed: int,
        use_cache: bool,
        completed_benchmark_prompts: list[BenchmarkingPrompt] | None = None,
        error_benchmark_prompts: list[BenchmarkingPrompt] | None = None,
        cancelled_benchmark_prompts: list[BenchmarkingPrompt] | None = None,
        completed_benchmark_metrics: list[str] | None = None,
        error_benchmark_metrics: list[str] | None = None,
        cancelled_benchmark_metrics: list[str] | None = None,
        num_of_total_metrics: AtomicInteger = AtomicInteger(0),
        num_of_completed_metrics: AtomicInteger = AtomicInteger(0),
        num_of_error_metrics: AtomicInteger = AtomicInteger(0),
        num_of_cancelled_metrics: AtomicInteger = AtomicInteger(0),
        num_of_total_prompts: AtomicInteger = AtomicInteger(0),
        num_of_completed_prompts: AtomicInteger = AtomicInteger(0),
        num_of_error_prompts: AtomicInteger = AtomicInteger(0),
        num_of_cancelled_prompts: AtomicInteger = AtomicInteger(0),
        start_time: datetime = datetime.now(),
        end_time: datetime = datetime.now(),
        results: dict | None = None,
    ):
        self.database_instance = database_instance
        self.recipe_name = recipe_name
        self.recipe_instance = recipe_instance
        self.connector = connector
        self.prompt_selection_percentage = prompt_selection_percentage
        self.random_seed = random_seed
        self.use_cache = use_cache
        self.completed_benchmark_prompts = (
            completed_benchmark_prompts
            if completed_benchmark_prompts is not None
            else []
        )
        self.error_benchmark_prompts = (
            error_benchmark_prompts if error_benchmark_prompts is not None else []
        )
        self.cancelled_benchmark_prompts = (
            cancelled_benchmark_prompts
            if cancelled_benchmark_prompts is not None
            else []
        )
        self.completed_benchmark_metrics = (
            completed_benchmark_metrics
            if completed_benchmark_metrics is not None
            else []
        )
        self.error_benchmark_metrics = (
            error_benchmark_metrics if error_benchmark_metrics is not None else []
        )
        self.cancelled_benchmark_metrics = (
            cancelled_benchmark_metrics
            if cancelled_benchmark_metrics is not None
            else []
        )
        self.num_of_total_metrics = num_of_total_metrics
        self.num_of_completed_metrics = num_of_completed_metrics
        self.num_of_error_metrics = num_of_error_metrics
        self.num_of_cancelled_metrics = num_of_cancelled_metrics
        self.num_of_total_prompts = num_of_total_prompts
        self.num_of_completed_prompts = num_of_completed_prompts
        self.num_of_error_prompts = num_of_error_prompts
        self.num_of_cancelled_prompts = num_of_cancelled_prompts
        self.start_time = start_time
        self.end_time = end_time
        self.prompts_generator = None
        self.results = results if results is not None else {}

    class Config:
        arbitrary_types_allowed = True

    async def execute_llm_query(
        self,
        cancel_event: asyncio.Event,
        progress_callback_fn: Callable,
        completed_query_llm_queue: asyncio.Queue,
    ) -> None:
        """
        Executes LLM queries for each benchmarking prompt in the prompts generator.

        This method processes each benchmarking prompt asynchronously, checking for cached results
        and performing predictions if necessary. It updates the status of each prompt based on the
        outcome and appends it to the appropriate list.

        Args:
            cancel_event (asyncio.Event): An event to signal cancellation of the process.
            completed_query_llm_queue (asyncio.Queue): A queue to store completed tasks for further processing.

        Raises:
            RuntimeError: If the prompts generator is not set.
        """
        if self.prompts_generator:
            self.start_time = datetime.now()
            self.end_time = datetime.now()

            async def query_benchmark_prompt(
                benchmark_prompt: BenchmarkingPrompt,
            ) -> None:
                """
                Queries a single benchmarking prompt, checking cache and performing prediction if needed.

                Args:
                    benchmark_prompt (BenchmarkingPrompt): The benchmarking prompt to be processed.
                """
                if cancel_event.is_set():
                    logger.warning(BENCHMARKING_CANCEL_SET_WARNING)
                    await self.num_of_cancelled_prompts.increment()
                    self.cancelled_benchmark_prompts.append(benchmark_prompt)
                    return

                cache_record = None
                if self.use_cache:
                    try:
                        cache_record = Storage.read_database_record(
                            self.database_instance,
                            (
                                benchmark_prompt.connection_id,
                                benchmark_prompt.recipe_id,
                                benchmark_prompt.dataset_id,
                                benchmark_prompt.prompt_template_id,
                                benchmark_prompt.connector_prompt.prompt,
                            ),
                            self.sql_read_runner_cache_record,
                        )
                    except Exception:
                        cache_record = None

                # If cache record does not exist, perform prediction and cache the result
                if cache_record is None:
                    try:
                        benchmark_prompt.connector_prompt = (
                            await Connector.get_prediction(
                                benchmark_prompt.connector_prompt, self.connector
                            )
                        )
                        Storage.create_database_record(
                            self.database_instance,
                            benchmark_prompt.to_tuple(),
                            self.sql_create_runner_cache_record,
                        )
                        await self.num_of_completed_prompts.increment()
                        self.completed_benchmark_prompts.append(benchmark_prompt)
                    except Exception:
                        await self.num_of_error_prompts.increment()
                        self.error_benchmark_prompts.append(benchmark_prompt)
                else:
                    # Load result from cache
                    benchmark_prompt = BenchmarkingPrompt.from_tuple(cache_record)
                    await self.num_of_completed_prompts.increment()
                    self.completed_benchmark_prompts.append(benchmark_prompt)

                # Provide progress update
                await progress_callback_fn()

            # Create a list of tasks for processing prompts
            asyncio_tasks = [
                asyncio.create_task(query_benchmark_prompt(benchmark_prompt))
                async for benchmark_prompt in self.prompts_generator
            ]

            # Wait for all tasks to complete
            await asyncio.gather(*asyncio_tasks)

            # After processing, put the task into the completed_query_llm_queue
            await completed_query_llm_queue.put(self)
        else:
            raise RuntimeError(BENCHMARKINGTASK_EXECUTE_LLM_QUERY_ERROR_MSG)

    async def execute_metrics_calculation(
        self, cancel_event: asyncio.Event, progress_callback_fn: Callable
    ) -> None:
        # Sort the predictions into groups for prompt templates
        grouped_prompt_template_preds = {}
        try:
            # Sort completed prompts by prompt_template_id
            self.completed_benchmark_prompts.sort(
                key=attrgetter(
                    "connection_id", "recipe_id", "dataset_id", "prompt_template_id"
                )
            )

            # Group prompts by prompt_template_id
            for key, group in groupby(
                self.completed_benchmark_prompts,
                key=attrgetter(
                    "connection_id", "recipe_id", "dataset_id", "prompt_template_id"
                ),
            ):
                group_list = list(group)
                grouped_prompt_template_preds[key] = {
                    "prompts": [pred.connector_prompt.prompt for pred in group_list],
                    "predicted_results": [
                        pred.connector_prompt.predicted_results for pred in group_list
                    ],
                    "targets": [pred.connector_prompt.target for pred in group_list],
                    "durations": [
                        pred.connector_prompt.duration for pred in group_list
                    ],
                }

        except Exception as e:
            logger.error(f"Error while grouping prompt templates: {str(e)}")

        # Generate metrics results
        for (
            group_recipe_key,
            group_recipe_value,
        ) in grouped_prompt_template_preds.items():
            metrics_result = []
            prompts = group_recipe_value["prompts"]
            predicted_results = group_recipe_value["predicted_results"]
            targets = group_recipe_value["targets"]

            for metric_name in self.recipe_instance.metrics:
                if cancel_event.is_set():
                    logger.warning(BENCHMARKING_CANCEL_SET_WARNING)
                    await self.num_of_cancelled_metrics.increment()
                    self.cancelled_benchmark_metrics.append(metric_name)
                    return

                try:
                    logger.info(
                        BENCHMARKINGTASK_EXECUTE_METRICS_CALCULATION_MSG.format(
                            metric_name=metric_name,
                            num_of_completed_prompts=await self.num_of_completed_prompts.get(),
                        )
                    )

                    # Load the metric and run
                    metric_instance = Metric.load(metric_name)
                    metrics_result.append(
                        await metric_instance.get_results(prompts, predicted_results, targets)  # type: ignore ; ducktyping # noqa: E501
                    )

                    # Format the results to have data and metrics results
                    group_data = []
                    durations = group_recipe_value["durations"]
                    for prompt, predicted_result, target, duration in zip(
                        prompts, predicted_results, targets, durations
                    ):
                        group_data.append(
                            {
                                "prompt": prompt,
                                "predicted_result": predicted_result.to_dict(),
                                "target": target,
                                "duration": duration,
                            }
                        )

                    # Append results for recipe
                    self.results[group_recipe_key] = {
                        "data": group_data,
                        "results": metrics_result,
                    }

                    await self.num_of_completed_metrics.increment()
                    self.completed_benchmark_metrics.append(metric_name)

                except Exception:
                    await self.num_of_error_metrics.increment()
                    self.error_benchmark_metrics.append(metric_name)

                # Provide progress update
                await progress_callback_fn()

    async def setup(self) -> None:
        """
        Initialize the benchmarking task by generating prompts asynchronously.

        This method prepares the prompts required for the benchmarking task
        using the associated recipe instance and templates, and updates the
        total number of prompts and metrics.

        Raises:
            RuntimeError: If the recipe instance is not set or metrics cannot be determined.
        """
        try:
            num_of_prompts, prompts_generator = await self.create_prompts()
            self.prompts_generator = prompts_generator
            await self.num_of_total_prompts.set(num_of_prompts)
            await self.num_of_total_metrics.set(len(self.recipe_instance.metrics))

        except Exception as e:
            raise RuntimeError(BENCHMARKINGTASK_SETUP_ERROR.format(message=str(e)))

    async def calculate_progress(self) -> float:
        """
        Calculate the progress of the benchmarking task.

        This method calculates the progress based on the number of completed, error, and cancelled prompts
        relative to the total number of prompts. This accounts for 50% of the total progress. The other 50%
        comes from the metrics.

        Returns:
            float: The progress as a percentage (0.0 to 100.0).
        """
        total_processed_prompts = (
            await self.num_of_completed_prompts.get()
            + await self.num_of_error_prompts.get()
            + await self.num_of_cancelled_prompts.get()
        )
        if await self.num_of_total_prompts.get() == 0:
            prompt_progress = 0.0
        else:
            prompt_progress = (
                total_processed_prompts / await self.num_of_total_prompts.get()
            ) * 50.0

        total_processed_metrics = await self.num_of_completed_metrics.get()
        if await self.num_of_total_metrics.get() == 0:
            metric_progress = 0.0
        else:
            metric_progress = (
                total_processed_metrics / await self.num_of_total_metrics.get()
            ) * 50.0

        return prompt_progress + metric_progress

    async def create_prompts(
        self,
    ) -> tuple[int, AsyncGenerator[BenchmarkingPrompt, None]]:
        """
        Asynchronously creates prompts and returns the total count along with an asynchronous generator.

        This method retrieves prompt templates from storage, calculates the total number of prompts
        that will be generated, and returns both the total count and an asynchronous generator
        for the prompts.

        Returns:
            tuple[int, AsyncGenerator[BenchmarkingPrompt, None]]: A tuple containing the total count
            of prompts and an asynchronous generator for generating the prompts.
        """
        pt_id = "no-template"
        templates: dict[str, str] = {}
        if self.recipe_instance and self.recipe_instance.prompt_templates:
            for pt_id in self.recipe_instance.prompt_templates:
                pt_info = Storage.read_object(
                    EnvVariables.PROMPT_TEMPLATES.name, pt_id, "json"
                )
                templates[pt_id] = pt_info["template"]

        # Calculate the total number of prompts that will be generated
        # Setting to_log=False to avoid logging during the counting process
        count = 0
        async for _ in self.create_prompts_generator(templates, to_log=False):
            count += 1

        # Return the total count and the generator
        return count, self.create_prompts_generator(templates)

    async def create_prompts_generator(
        self, templates: dict[str, str], to_log: bool = True
    ) -> AsyncGenerator[BenchmarkingPrompt, None]:
        """
        Asynchronously generates prompts using the provided templates or yields original prompts
        if no templates are available.

        This method iterates over datasets and applies templates to render prompts.
        If no templates are available, it yields the original prompts from the datasets.

        Args:
            templates (dict[str, str]): A dictionary mapping template IDs to template strings.
            to_log (bool): A flag indicating whether to log the number of selected prompts.

        Yields:
            BenchmarkingPrompt: An instance of BenchmarkingPrompt containing all necessary information for
            processing the prompt.
        """
        for ds_id in self.recipe_instance.datasets:
            async for prompt_index, prompt in self.get_dataset_prompts(ds_id, to_log):
                if templates:
                    for pt_id, pt_template in templates.items():
                        actual_prompt = Template(pt_template).render(
                            {"prompt": prompt["input"]}
                        )
                        yield BenchmarkingPrompt(
                            connection_id=self.connector.id,
                            recipe_id=self.recipe_instance.id,
                            dataset_id=ds_id,
                            prompt_template_id=pt_id,
                            connector_prompt=ConnectorPromptArguments(
                                prompt_index=prompt_index,
                                prompt=actual_prompt,
                                target=prompt["target"],
                            ),
                        )
                else:
                    yield BenchmarkingPrompt(
                        connection_id=self.connector.id,
                        recipe_id=self.recipe_instance.id,
                        dataset_id=ds_id,
                        prompt_template_id=pt_id,
                        connector_prompt=ConnectorPromptArguments(
                            prompt_index=prompt_index,
                            prompt=prompt["input"],
                            target=prompt["target"],
                        ),
                    )

    async def get_dataset_prompts(
        self, ds_id: str, to_log: bool
    ) -> AsyncGenerator[tuple[int, dict[str, str]], None]:
        """
        Asynchronously retrieves prompts from a dataset using the specified dataset ID.

        This method selects prompts based on the prompt_selection_percentage and random_seed.
        It yields each selected prompt along with its index.

        Args:
            ds_id (str): The unique identifier of the dataset from which to retrieve prompts.
            to_log (bool): A flag indicating whether to log the number of selected prompts.

        Yields:
            tuple[int, dict[str, str]]: A tuple containing the index of the prompt and the prompt data.
        """
        # Retrieve dataset arguments
        ds_args = Dataset.read(ds_id)

        if ds_args.num_of_dataset_prompts == 0:
            prompt_indices = []
        else:
            # Calculate the number of prompts to select based on prompt_selection_percentage
            self.num_of_prompts = max(
                1,
                int(
                    (self.prompt_selection_percentage / 100)
                    * ds_args.num_of_dataset_prompts
                ),
            )
            if self.num_of_prompts == ds_args.num_of_dataset_prompts:
                prompt_indices = range(ds_args.num_of_dataset_prompts)
            else:
                random.seed(self.random_seed)
                prompt_indices = random.sample(
                    range(ds_args.num_of_dataset_prompts), self.num_of_prompts
                )

        if to_log:
            logger.debug(
                BENCHMARKINGTASK_GET_DATASET_PROMPTS_MSG.format(
                    ds_id=ds_id,
                    num_of_prompts=len(prompt_indices),
                    total_dataset_prompts=ds_args.num_of_dataset_prompts,
                )
            )

        # Iterate over the dataset examples and yield prompts based on the generated indices
        for prompts_gen_index, prompts_data in enumerate(ds_args.examples):
            if prompts_gen_index in prompt_indices:
                yield prompts_gen_index, prompts_data

    async def get_progress_update(self) -> dict[str, int | list[BenchmarkingPrompt]]:
        """
        Asynchronously retrieves the current progress update of the benchmarking task.

        This method returns a dictionary containing the total and completed metrics and prompts,
        the number of error and cancelled prompts, and the lists of completed, error, and cancelled prompts.
        It also includes the overall progress percentage calculated asynchronously.

        Returns:
            dict[str, int | list[BenchmarkingPrompt]]: A dictionary with keys representing different progress
            metrics and their corresponding integer values, as well as lists of completed, error, and cancelled prompts.
        """
        return {
            "num_of_total_metrics": await self.num_of_total_metrics.get(),
            "num_of_completed_metrics": await self.num_of_completed_metrics.get(),
            "num_of_total_prompts": await self.num_of_total_prompts.get(),
            "num_of_completed_prompts": await self.num_of_completed_prompts.get(),
            "num_of_error_prompts": await self.num_of_error_prompts.get(),
            "num_of_cancelled_prompts": await self.num_of_cancelled_prompts.get(),
            "completed_prompts": self.completed_benchmark_prompts,
            "error_prompts": self.error_benchmark_prompts,
            "cancelled_prompts": self.cancelled_benchmark_prompts,
            "progress": int(await self.calculate_progress()),
        }


# -----------------------------------------------
# Benchmarking Class
# -----------------------------------------------
class Benchmarking:
    async def handle_task_progress(
        self,
        benchmarking_tasks: list[BenchmarkingTask],
        run_progress: RunProgress,
        num_of_completed_prompts_threshold: int,
        num_of_cancelled_prompts_threshold: int,
        num_of_error_prompts_threshold: int,
    ) -> None:
        """
        Handle and update the progress of benchmarking tasks asynchronously.

        This method aggregates progress data from a list of benchmarking tasks, including metrics and prompts,
        and updates the run progress. It also checks against specified thresholds for completed, cancelled,
        and error prompts.

        Args:
            benchmarking_tasks (list[BenchmarkingTask]): A list of benchmarking tasks to aggregate progress from.
            run_progress (RunProgress): The RunProgress instance to update with the aggregated progress data.
            num_of_completed_prompts_threshold (int): Threshold for the number of completed prompts.
            num_of_cancelled_prompts_threshold (int): Threshold for the number of cancelled prompts.
            num_of_error_prompts_threshold (int): Threshold for the number of error prompts.
        """
        aggregated_result = {
            "num_of_total_metrics": 0,
            "num_of_completed_metrics": 0,
            "num_of_total_prompts": 0,
            "num_of_completed_prompts": 0,
            "num_of_error_prompts": 0,
            "num_of_cancelled_prompts": 0,
            "current_progress": 0,
            "current_prompt_progress": 0,
            "current_metric_progress": 0,
            "completed_prompts": [],
            "error_prompts": [],
            "cancelled_prompts": [],
        }

        for task in benchmarking_tasks:
            result = await task.get_progress_update()
            aggregated_result["num_of_total_metrics"] += result["num_of_total_metrics"]
            aggregated_result["num_of_completed_metrics"] += result[
                "num_of_completed_metrics"
            ]
            aggregated_result["num_of_total_prompts"] += result["num_of_total_prompts"]
            aggregated_result["num_of_completed_prompts"] += result[
                "num_of_completed_prompts"
            ]
            aggregated_result["num_of_error_prompts"] += result["num_of_error_prompts"]
            aggregated_result["num_of_cancelled_prompts"] += result[
                "num_of_cancelled_prompts"
            ]
            aggregated_result["current_progress"] += result["progress"]
            aggregated_result["completed_prompts"].extend(result["completed_prompts"])
            aggregated_result["error_prompts"].extend(result["error_prompts"])
            aggregated_result["cancelled_prompts"].extend(result["cancelled_prompts"])

        if benchmarking_tasks:
            aggregated_result["current_progress"] = int(
                aggregated_result["current_progress"] / len(benchmarking_tasks)
            )
            if aggregated_result["num_of_total_prompts"] > 0:
                aggregated_result["current_prompt_progress"] = int(
                    (
                        aggregated_result["num_of_completed_prompts"]
                        / aggregated_result["num_of_total_prompts"]
                    )
                    * 100
                )
            if aggregated_result["num_of_total_metrics"] > 0:
                aggregated_result["current_metric_progress"] = int(
                    (
                        aggregated_result["num_of_completed_metrics"]
                        / aggregated_result["num_of_total_metrics"]
                    )
                    * 100
                )

        run_progress.notify_progress(
            total_num_of_tasks=len(benchmarking_tasks),
            total_num_of_prompts=aggregated_result["num_of_total_prompts"],
            total_num_of_metrics=aggregated_result["num_of_total_metrics"],
            completed_num_of_prompts=aggregated_result["num_of_completed_prompts"],
            cancelled_num_of_prompts=aggregated_result["num_of_cancelled_prompts"],
            error_num_of_prompts=aggregated_result["num_of_error_prompts"],
            completed_num_of_metrics=aggregated_result["num_of_completed_metrics"],
            overall_prompt_progress=aggregated_result["current_prompt_progress"],
            overall_metric_progress=aggregated_result["current_metric_progress"],
            overall_progress=aggregated_result["current_progress"],
            completed_prompts=aggregated_result["completed_prompts"][
                :num_of_completed_prompts_threshold
            ],
            error_prompts=aggregated_result["error_prompts"][
                :num_of_error_prompts_threshold
            ],
            cancelled_prompts=aggregated_result["cancelled_prompts"][
                :num_of_cancelled_prompts_threshold
            ],
        )

    async def completed_query_llm_handler(
        self,
        completed_query_llm_queue: asyncio.Queue[BenchmarkingTask],
        metrics_calculation_queue: asyncio.Queue[BenchmarkingTask],
    ) -> None:
        """
        Continuously processes tasks from the completed query LLM queue and transfers them to the
        metrics calculation queue.

        This method operates in an infinite loop, checking for tasks in the completed_query_llm_queue.
        If the cancel_event is triggered, it logs a warning and exits the loop.

        Otherwise, it retrieves each task from the completed_query_llm_queue and places it into the
        metrics_calculation_queue for further processing.

        Args:
            completed_query_llm_queue (asyncio.Queue[BenchmarkingTask]): The queue from which completed tasks
                                                                         are retrieved for LLM querying.
            metrics_calculation_queue (asyncio.Queue[BenchmarkingTask]): The queue to which tasks are added for
                                                                  metrics calculation.
        """
        while True:
            if self.cancel_event.is_set():
                logger.warning(BENCHMARKING_CANCEL_SET_WARNING)
                break

            task = await completed_query_llm_queue.get()
            await metrics_calculation_queue.put(task)

            # Check if it's the sentinel, which signals the end
            if task is None:
                completed_query_llm_queue.task_done()  # Mark task as done before breaking
                break

            completed_query_llm_queue.task_done()

    async def metrics_calculation_handler(
        self, metrics_calculation_queue: asyncio.Queue[BenchmarkingTask]
    ) -> None:
        """
        Continuously processes tasks from the metrics calculation queue.

        This method runs an infinite loop to check for tasks in the metrics_calculation_queue.

        If the cancel_event is set, it logs a warning and breaks the loop.
        Otherwise, it retrieves each task from the metrics_calculation_queue and processes it by generating
        metrics results.

        Args:
            metrics_calculation_queue (asyncio.Queue[BenchmarkingTask]): The queue from which tasks are retrieved
            for metrics calculation.
        """
        while True:
            if self.cancel_event.is_set():
                logger.warning(BENCHMARKING_CANCEL_SET_WARNING)
                break

            task = await metrics_calculation_queue.get()
            # Check if it's the sentinel, which signals the end
            if task is None:
                metrics_calculation_queue.task_done()
                break

            # Process the task
            await task.execute_metrics_calculation(
                self.cancel_event,
                lambda: self.handle_task_progress(
                    [task],
                    self.run_progress,
                    self.num_of_completed_prompts_threshold,
                    self.num_of_cancelled_prompts_threshold,
                    self.num_of_error_prompts_threshold,
                ),
            )
            metrics_calculation_queue.task_done()

    async def generate(
        self,
        event_loop: asyncio.AbstractEventLoop,
        run_id: str,
        runner_id: str,
        runner_args: dict,
        database_instance: DBInterface | None,
        endpoints: list[str],
        run_progress: RunProgress,
        cancel_event: asyncio.Event,
    ) -> ResultArguments | None:
        """
        Asynchronously generates benchmarking results based on the provided runner arguments and
        stores them in the database.

        This method manages the benchmarking process by setting up the environment, executing the recipes and
        cookbooks, and gathering the results.

        It uses the provided database instance to cache and retrieve runner data.

        Args:
            event_loop (asyncio.AbstractEventLoop): The event loop for scheduling asynchronous tasks.
            run_id (str): The unique identifier for the run.
            runner_id (str): The unique identifier for the runner.
            runner_args (dict): A dictionary containing arguments for the runner.
            database_instance (DBInterface | None): The database interface for storing and retrieving runner data.
            endpoints (list[str]): A list of endpoint identifiers for the benchmarking process.
            run_progress (RunProgress): An object to report the progress of the run.
            cancel_event (asyncio.Event): An event to signal cancellation of the process.

        Returns:
            ResultArguments | None: The result arguments object containing the results of the benchmarking process,
            or None if the process is cancelled or fails to generate results.
        """
        try:
            if not database_instance:
                run_progress.notify_error(
                    BENCHMARKING_GENERATE_NO_DATABASE_INSTANCE_MSG
                )
                raise RuntimeError(BENCHMARKING_GENERATE_NO_DATABASE_INSTANCE_MSG)

            # Store parsed values
            self.run_id = run_id
            self.runner_id = runner_id
            self.runner_args = runner_args
            self.database_instance = database_instance
            self.endpoints = endpoints
            self.run_progress = run_progress
            self.cancel_event = cancel_event

            # Log the assignment of parameters using BENCHMARKING_RUN_DEBUG_MSG
            logger.debug(
                BENCHMARKING_RUN_DEBUG_MSG.format(
                    run_id=run_id,
                    runner_id=runner_id,
                    runner_args=runner_args,
                    database_instance=database_instance,
                    endpoints=endpoints,
                    run_progress=run_progress,
                    cancel_event=cancel_event,
                )
            )

            # Run benchmarking
            await self.run()

        except Exception as e:
            # Log the error message if an exception occurs
            run_progress.notify_error(
                BENCHMARKING_GENERATE_ERROR.format(message=str(e))
            )

        finally:
            # Log the update of benchmarking status
            logger.debug(BENCHMARKING_GENERATE_UPDATE_BENCHMARK_STATUS)
            if cancel_event.is_set():
                # Notify progress as cancelled if the cancel event is set
                run_progress.notify_progress(
                    status=RunStatus.CANCELLED, total_progress=100
                )
            elif run_progress.run_arguments.error_messages:
                # Notify progress as completed with errors if there are error messages
                run_progress.notify_progress(
                    status=RunStatus.COMPLETED_WITH_ERRORS, total_progress=100
                )
            else:
                # Notify progress as completed if no errors occurred
                run_progress.notify_progress(
                    status=RunStatus.COMPLETED, total_progress=100
                )

    async def run(self) -> None:
        """
        Executes the benchmarking workflow, which includes:
        1. Retrieving necessary arguments from runner_args.
        2. Loading endpoint instances and configuring system prompts.
        3. Generating benchmarking tasks.
        4. Initiating task processing and monitoring completion.

        Raises:
            RuntimeError: If no endpoints are provided.

        Attributes:
            cookbooks (Optional[List[str]]): Names of cookbooks to be used.
            recipes (Optional[List[str]]): Names of recipes to be used.
            prompt_selection_percentage (int): Percentage of prompts to select.
            random_seed (int): Seed for random operations to ensure reproducibility.
            system_prompt (str): System prompt to configure.
            use_cache (bool): Flag to determine if caching should be used.
        """
        # Retrieve necessary arguments from runner_args
        self.cookbooks: list[str] | None = self.runner_args.get("cookbooks", None)
        self.num_of_cancelled_prompts_threshold: int = self.runner_args.get(
            "num_of_cancelled_prompts_threshold", 10
        )
        self.num_of_completed_prompts_threshold: int = self.runner_args.get(
            "num_of_completed_prompts_threshold", 10
        )
        self.num_of_error_prompts_threshold: int = self.runner_args.get(
            "num_of_error_prompts_threshold", 10
        )
        self.prompt_selection_percentage: int = self.runner_args.get(
            "prompt_selection_percentage", 100
        )
        self.random_seed: int = self.runner_args.get("random_seed", 0)
        self.recipes: list[str] | None = self.runner_args.get("recipes", None)
        self.system_prompt: str = self.runner_args.get("system_prompt", "")
        self.use_cache: bool = self.runner_args.get("use_cache", True)

        # Load endpoint instances and configure system prompts
        if self.endpoints:
            self.connectors = await self.get_connector_endpoint_instances(
                self.endpoints
            )
            logger.debug(BENCHMARKING_RUN_LOAD_ENDPOINT_INSTANCES_SUCCESS)
        else:
            raise RuntimeError(
                BENCHMARKING_RUN_LOAD_ENDPOINT_INSTANCES_NO_ENDPOINTS_MSG
            )

        if self.system_prompt:
            await self.set_system_prompts()
            logger.debug(BENCHMARKING_RUN_UPDATE_SYSTEM_PROMPT_SUCCESS)
        else:
            logger.warning(BENCHMARKING_RUN_UPDATE_SYSTEM_PROMPT_SKIPPED)

        # Generate benchmarking tasks
        (
            benchmark_tasks,
            benchmark_tasks_prompts,
            benchmark_tasks_metrics,
        ) = await self.generate_tasks()
        logger.info(
            BENCHMARKING_RUN_GENERATE_TASKS_INFO.format(
                message=len(benchmark_tasks),
                total_prompts=benchmark_tasks_prompts,
                total_metrics=benchmark_tasks_metrics,
            )
        )

        # Update progress information
        self.run_progress.notify_progress(
            total_num_of_tasks=len(benchmark_tasks),
            total_num_of_prompts=benchmark_tasks_prompts,
            total_num_of_metrics=benchmark_tasks_metrics,
        )

        # Check if there are benchmark tasks or prompts for processing
        if (
            not benchmark_tasks
            or benchmark_tasks_prompts == 0
            or benchmark_tasks_metrics == 0
        ):
            logger.warning(BENCHMARKING_RUN_CHECK_TASKS_PROMPTS_METRICS)
            return

        # Check if the cancel event is set, and if so, log a warning and exit the function
        if self.cancel_event.is_set():
            logger.warning(BENCHMARKING_CANCEL_SET_WARNING)
            return

        # Create queues for task processing
        completed_query_llm_queue = asyncio.Queue()  # Queue for external LLM completion
        metrics_calculation_queue = asyncio.Queue()  # Queue for metrics calculation

        # Start task processing
        asyncio.create_task(
            self.completed_query_llm_handler(
                completed_query_llm_queue, metrics_calculation_queue
            )
        )
        asyncio.create_task(self.metrics_calculation_handler(metrics_calculation_queue))

        # Wait for all tasks to complete
        logger.info(
            BENCHMARKING_RUN_QUERY_LLM_INFO.format(num_of_tasks=len(benchmark_tasks))
        )
        asyncio_tasks = [
            task.execute_llm_query(
                self.cancel_event,
                lambda: self.handle_task_progress(
                    benchmark_tasks,
                    self.run_progress,
                    self.num_of_completed_prompts_threshold,
                    self.num_of_cancelled_prompts_threshold,
                    self.num_of_error_prompts_threshold,
                ),
                completed_query_llm_queue,
            )
            for task in benchmark_tasks
        ]
        await asyncio.gather(*asyncio_tasks)

        # Signal completion of task processing
        await completed_query_llm_queue.put(None)

        # Wait until all tasks in the queues are processed
        await completed_query_llm_queue.join()
        await metrics_calculation_queue.join()

        # Combine all the results into recipe or cookbook

    async def generate_tasks(self) -> tuple[list[BenchmarkingTask], int, int]:
        """
        Asynchronously generates benchmarking tasks using the specified cookbooks and recipes.

        This method iterates over each recipe in the provided cookbooks and standalone recipes,
        creating `BenchmarkingTask` instances for each. The tasks are collected into a list,
        which is returned along with the total number of prompts and metrics.

        Returns:
            tuple: A tuple containing:
                - A list of `BenchmarkingTask` instances representing the tasks to be executed.
                - An integer representing the total number of prompts across all tasks.
                - An integer representing the total number of metrics across all tasks.

        Raises:
            RuntimeError: If an error occurs while loading a cookbook or recipe, a RuntimeError is raised with
            a descriptive message.
        """
        tasks: list[BenchmarkingTask] = []
        total_number_of_tasks_prompts: int = 0
        total_number_of_tasks_metrics: int = 0

        if self.cookbooks:
            for cookbook in self.cookbooks:
                try:
                    cookbook_instance = Cookbook.load(cookbook)
                    for recipe in cookbook_instance.recipes:
                        (
                            new_tasks,
                            new_tasks_num_of_prompts,
                            new_tasks_num_of_metrics,
                        ) = await self.generate_recipe_task(recipe)
                        tasks.extend(new_tasks)
                        total_number_of_tasks_prompts += new_tasks_num_of_prompts
                        total_number_of_tasks_metrics += new_tasks_num_of_metrics

                except Exception as e:
                    raise RuntimeError(
                        BENCHMARKING_GENERATE_TASKS_COOKBOOK_ERROR_MSG.format(
                            cookbook_name=cookbook, message=str(e)
                        )
                    )

        if self.recipes:
            for recipe in self.recipes:
                try:
                    (
                        new_tasks,
                        new_tasks_num_of_prompts,
                        new_tasks_num_of_metrics,
                    ) = await self.generate_recipe_task(recipe)
                    tasks.extend(new_tasks)
                    total_number_of_tasks_prompts += new_tasks_num_of_prompts
                    total_number_of_tasks_metrics += new_tasks_num_of_metrics

                except Exception as e:
                    raise RuntimeError(
                        BENCHMARKING_GENERATE_TASKS_RECIPE_ERROR_MSG.format(
                            recipe_name=recipe, message=str(e)
                        )
                    )

        return tasks, total_number_of_tasks_prompts, total_number_of_tasks_metrics

    async def generate_recipe_task(
        self, recipe: str
    ) -> tuple[list[BenchmarkingTask], int, int]:
        """
        Asynchronously generates benchmarking tasks for a given recipe.

        This method loads the specified recipe and iterates over each connector to create
        `BenchmarkingTask` instances. It initializes each task with the necessary attributes.

        Args:
            recipe (str): The name of the recipe to load and generate tasks for.

        Returns:
            tuple[list[BenchmarkingTask], int, int]: A tuple containing:
                - A list of generated `BenchmarkingTask` instances.
                - The total number of prompts across all tasks.
                - The total number of metrics across all tasks.
        """
        new_tasks: list[BenchmarkingTask] = []
        new_tasks_num_of_prompts: int = 0
        new_tasks_num_of_metrics: int = 0
        recipe_instance = Recipe.load(recipe)

        # Loop through each connector and create a task for each
        for connector in self.connectors:
            new_task = BenchmarkingTask(
                self.database_instance,
                recipe,
                recipe_instance,
                connector,
                self.prompt_selection_percentage,
                self.random_seed,
                self.use_cache,
            )
            await new_task.setup()

            new_tasks.append(new_task)
            new_tasks_num_of_prompts += await new_task.num_of_total_prompts.get()
            new_tasks_num_of_metrics += await new_task.num_of_total_metrics.get()

        return new_tasks, new_tasks_num_of_prompts, new_tasks_num_of_metrics

    async def get_connector_endpoint_instances(
        self, endpoints: list[str]
    ) -> list[Connector]:
        """
        Asynchronously loads connector endpoint instances from a list of endpoint identifiers.

        Args:
            endpoints (list[str]): A list of endpoint identifiers to be loaded.

        Returns:
            list[Connector]: A list of Connector instances created from the specified endpoints.

        Raises:
            RuntimeError: If any endpoint fails to load, an error message is logged, and the exception is raised.
        """
        connectors = []
        for endpoint in endpoints:
            try:
                connectors.append(Connector.create(ConnectorEndpoint.read(endpoint)))
            except Exception as e:
                raise RuntimeError(
                    BENCHMARKING_GET_CONNECTOR_ENDPOINT_INSTANCES_LOAD_FAILED_MSG.format(
                        endpoint_name=endpoint, message=str(e)
                    )
                )
        return connectors

    async def set_system_prompts(self) -> None:
        """
        Asynchronously sets the system prompt for each connector in the instance's connectors list.

        Raises:
            RuntimeError: If setting the system prompt fails for any connector,
                          an error message is logged and the exception is raised.
        """
        for connector in self.connectors:
            try:
                connector.set_system_prompt(self.system_prompt)
            except Exception as e:
                raise RuntimeError(
                    BENCHMARKING_SET_SYSTEM_PROMPTS_FAILED_MSG.format(
                        connector_name=connector.id, message=str(e)
                    )
                )
