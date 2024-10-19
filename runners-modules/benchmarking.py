from __future__ import annotations

import ast
import asyncio
import json
import random
from datetime import datetime
from typing import Any, AsyncGenerator

from jinja2 import Template
from moonshot.src.configs.env_variables import EnvVariables
from moonshot.src.connectors.connector import Connector
from moonshot.src.connectors.connector_prompt_arguments import ConnectorPromptArguments
from moonshot.src.connectors.connector_response import ConnectorResponse
from moonshot.src.connectors_endpoints.connector_endpoint import ConnectorEndpoint
from moonshot.src.cookbooks.cookbook import Cookbook
from moonshot.src.datasets.dataset import Dataset
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
# Benchmarking Task Manager Messages
# ---------------------------------------------------------------------------------------------------------------------
BENCHMARKING_TASK_MANAGER_TASK_INFO_MSG = "[BenchmarkingTaskManager] Recipe Task {recipe_task} for connector {connector_id} is at {progress}%."  # noqa: E501
BENCHMARKING_TASK_MANAGER_OVERALL_PROGRESS_INFO_MSG = (
    "[BenchmarkingTaskManager] Overall task progress is at {progress}%."
)
BENCHMARKING_TASK_MANAGER_NOTIFY_ERROR_MSG = (
    "[BenchmarkingTaskManager] Encountered error(s) during benchmarking."
)
# ---------------------------------------------------------------------------------------------------------------------
# Benchmarking Messages
# ---------------------------------------------------------------------------------------------------------------------
BENCHMARKING_GENERATE_NO_DATABASE_INSTANCE_MSG = (
    "No database instance provided. Exiting."
)
BENCHMARKING_RUN_WORKFLOW_LOAD_ENDPOINT_INSTANCES_NO_ENDPOINTS_MSG = (
    "No connector endpoints provided. Exiting."
)
BENCHMARKING_GENERATE_TASKS_COOKBOOK_ERROR_MSG = (
    "Failed to load cookbook '{cookbook_name}': {message}. Exiting."
)
BENCHMARKING_GENERATE_TASKS_RECIPE_ERROR_MSG = (
    "Failed to load recipe '{recipe_name}': {message}. Exiting."
)
BENCHMARKING_GET_CONNECTOR_ENDPOINT_INSTANCES_LOAD_FAILED_MSG = (
    "Failed to load '{endpoint_name}': {message}. Exiting."
)
BENCHMARKING_SET_SYSTEM_PROMPTS_FAILED_MSG = (
    "Failed to set system prompt on '{connector_name}': {message}. Exiting."
)
BENCHMARKING_GENERATE_ERROR = (
    "[Benchmarking] Failed to generate benchmarking results: {message}"
)
BENCHMARKING_GENERATE_UPDATE_BENCHMARK_STATUS = (
    "[Benchmarking] Updating benchmarking status."
)
BENCHMARKING_RUN_WORKFLOW_LOAD_ENDPOINT_INSTANCES_SUCCESS = (
    "[Benchmarking] Loaded connector endpoints."
)
BENCHMARKING_RUN_WORKFLOW_UPDATE_SYSTEM_PROMPT_SUCCESS = (
    "[Benchmarking] Updated connector system prompts."
)
BENCHMARKING_RUN_WORKFLOW_UPDATE_SYSTEM_PROMPT_SKIPPED = (
    "[Benchmarking] Skipped updating connector system prompts."
)
BENCHMARKING_RUN_WORKFLOW_GENERATE_TASKS_INFO = "[Benchmarking] Generated {message} tasks containing a total of {total_prompts} prompts."  # noqa: E501
BENCHMARKING_RUN_WORKFLOW_QUERY_LLM_INFO = (
    "[Benchmarking] Querying LLMs for {num_of_tasks} task(s)."
)
BENCHMARKING_QUERY_LLM_HANDLER_CANCEL_SET_WARNING = (
    "[Benchmarking] Cancellation flag is set. Stopping query llm handler."
)
BENCHMARKING_METRICS_CALC_HANDLER_CANCEL_SET_WARNING = (
    "[Benchmarking] Cancellation flag is set. Stopping metrics calculation handler."
)
BENCHMARKING_QUERY_LLM_CANCEL_SET_WARNING = (
    "[Benchmarking] Cancellation flag is set. Stopping query llm."
)
BENCHMARKING_QUERY_LLM_TASK_DONE = "[Benchmarking] Task '{recipe_name}' for '{connector_id}' completed model querying in {time_taken}s for a total of {num_of_prompts} prompts. (Success: {num_completed_benchmark_prompts}, Cancelled: {num_cancelled_benchmark_prompts}, Error: {num_error_benchmark_prompts})"  # noqa: E501


# ---------------------------------------------------------------------------------------------------------------------
# Benchmarking Prompt Class
# ---------------------------------------------------------------------------------------------------------------------
class BenchmarkingPrompt(BaseModel):
    """
    Represents a benchmarking prompt with all necessary attributes for processing and evaluation.

    Attributes:
        conn_id (str): The ID of the connection, default is an empty string.
        rec_id (str): The ID of the recipe.
        ds_id (str): The ID of the dataset.
        pt_id (str): The ID of the prompt template.
        random_seed (int): The random seed used for generating deterministic results.
        system_prompt (str): The system-generated prompt used for benchmarking.
        attack_module_id (str): The attack module used for generating perturb prompts.
        connector_prompt (ConnectorPromptArguments): The prompt information to send.
    """

    conn_id: str = ""
    rec_id: str
    ds_id: str
    pt_id: str
    random_seed: int
    system_prompt: str
    attack_module_id: str
    connector_prompt: ConnectorPromptArguments

    def to_tuple(self) -> tuple:
        """
        Converts the BenchmarkingPrompt instance into a tuple.

        This method aggregates the attributes of the BenchmarkingPrompt instance into a tuple.
        The tuple is structured with the following attribute values in order:
        conn_id, rec_id, ds_id, pt_id, attack_module_id, prompt_index, prompt, target, predicted_results, duration,
        random_seed, and system_prompt.

        This ordered tuple is particularly useful for serialization purposes, such as storing the
        BenchmarkingPrompt data in a database or transmitting it across network boundaries.

        Returns:
            tuple: A tuple representation of the BenchmarkingPrompt instance.
        """
        return (
            self.conn_id,
            self.rec_id,
            self.ds_id,
            self.pt_id,
            self.attack_module_id,
            self.connector_prompt.prompt_index,
            self.connector_prompt.prompt,
            str(self.connector_prompt.target),
            json.dumps(
                self.connector_prompt.predicted_results.to_dict()
                if self.connector_prompt.predicted_results
                else {}
            ),
            str(self.connector_prompt.duration),
            self.random_seed,
            self.system_prompt,
        )

    def to_dict(self) -> dict:
        """
        Converts the BenchmarkingPrompt instance into a dictionary.

        This method aggregates the attributes of the BenchmarkingPrompt instance into a dictionary.
        The dictionary is structured with the following attribute values:
        conn_id, rec_id, ds_id, pt_id, attack_module_id, prompt_index, prompt, target, predicted_results, duration,
        random_seed, and system_prompt.

        This ordered dictionary is particularly useful for serialization purposes, such as storing the
        BenchmarkingPrompt data in a database or transmitting it across network boundaries.

        Returns:
            dict: A dictionary representation of the BenchmarkingPrompt instance.
        """
        return {
            "conn_id": self.conn_id,
            "rec_id": self.rec_id,
            "ds_id": self.ds_id,
            "pt_id": self.pt_id,
            "attack_module_id": self.attack_module_id,
            "prompt_index": self.connector_prompt.prompt_index,
            "prompt": self.connector_prompt.prompt,
            "target": str(self.connector_prompt.target),
            "predicted_results": json.dumps(
                self.connector_prompt.predicted_results.to_dict()
                if self.connector_prompt.predicted_results
                else {}
            ),
            "duration": str(self.connector_prompt.duration),
            "random_seed": self.random_seed,
            "system_prompt": self.system_prompt,
        }

    @classmethod
    def from_tuple(cls, cache_record: tuple) -> BenchmarkingPrompt:
        """
        Reconstitutes a BenchmarkingPrompt instance from a tuple representation.

        This method accepts a tuple with values that map to the attributes of a BenchmarkingPrompt object.
        The expected order of values in the tuple is:
        conn_id, rec_id, ds_id, pt_id, attack_module_id, prompt_index, prompt, target, predicted_results, duration,
        random_seed, and system_prompt. It constructs a new BenchmarkingPrompt instance using these values.
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
            conn_id=cache_record[1],
            rec_id=cache_record[2],
            ds_id=cache_record[3],
            pt_id=cache_record[4],
            attack_module_id=cache_record[6],
            connector_prompt=ConnectorPromptArguments(
                prompt_index=cache_record[7],
                prompt=cache_record[8],
                target=target,
                predicted_results=predicted_results,
                duration=float(cache_record[11]),
            ),
            random_seed=cache_record[12],
            system_prompt=cache_record[13],
        )


# ---------------------------------------------------------------------------------------------------------------------
# Benchmarking Task Manager Class
# ---------------------------------------------------------------------------------------------------------------------
class BenchmarkingTaskManager:
    """
    Manages multiple BenchmarkingTask instances and collates their progress and error information.

    Attributes:
        tasks (list[BenchmarkingTask]): A list of BenchmarkingTask instances being managed.
        total_num_of_tasks (AtomicInteger): The total number of benchmarking tasks.
        total_num_of_prompts (AtomicInteger): The total number of prompts across all tasks.
        num_of_errors_messages_limit (int): The limit for the number of error messages.
        num_cancelled_messages_limit (int): The limit for the number of cancelled messages.
        run_progress_cb (RunProgress): Callback for updating the run progress.
    """

    def __init__(
        self,
        run_progress_cb: RunProgress,
        error_messages_limit: int,
        cancelled_messages_limit: int,
    ):
        # The list of generated tasks
        self.tasks = []
        self.num_of_errors_messages_limit = error_messages_limit
        self.num_cancelled_messages_limit = cancelled_messages_limit

        # Total number of tasks and prompts
        self.total_num_of_tasks: AtomicInteger = AtomicInteger(0)
        self.total_num_of_prompts: AtomicInteger = AtomicInteger(0)

        # Run Callback function
        self.run_progress_cb = run_progress_cb

    async def register_new_task(self, new_task: BenchmarkingTask) -> None:
        """
        Registers a new benchmarking task and updates the total number of tasks and prompts.

        This method appends the new task to the list of tasks, increments the total number of tasks,
        and updates the total number of prompts based on the new task's total prompts.

        Args:
            new_task (BenchmarkingTask): The new benchmarking task to be registered.
        """
        self.tasks.append(new_task)
        await self.total_num_of_tasks.increment()
        await self.total_num_of_prompts.increment(
            await new_task.num_of_total_prompts.get()
        )

    async def notify_task_progress(self):
        """
        Notifies the progress of all benchmarking tasks.

        This method logs the status of all tasks, calculates the overall progress,
        and updates the run_progress_cb with the current progress.

        It also consolidates information on completed, error, and cancelled prompts,
        and updates the run progress callback with this information.
        """
        total_num_of_tasks = await self.total_num_of_tasks.get()

        # Consolidate all current task progress
        # Current information on prompts and overall progress
        current_num_of_completed_prompts = 0
        current_num_of_error_prompts = 0
        current_num_of_cancelled_prompts = 0
        current_task_progress = 0.0
        current_overall_progress = 0

        # Current list of cancelled or error benchmark prompts information
        current_cancelled_benchmark_prompts = []
        current_error_benchmark_prompts = []

        for task in self.tasks:
            task_progress = await task.progress.get()
            current_task_progress += task_progress

            # Store task completed, error, and cancelled prompts
            current_num_of_completed_prompts += (
                await task.num_of_completed_prompts.get()
            )
            current_num_of_error_prompts += await task.num_of_error_prompts.get()
            current_num_of_cancelled_prompts += (
                await task.num_of_cancelled_prompts.get()
            )

            # Extend with a limit based on num_cancelled_messages_limit and num_of_errors_messages_limit
            current_cancelled_benchmark_prompts.extend(
                task.cancelled_benchmark_prompts[
                    : self.num_cancelled_messages_limit
                    - len(current_cancelled_benchmark_prompts)
                ]
            )
            current_error_benchmark_prompts.extend(
                [
                    prompt.to_dict()
                    for prompt in task.error_benchmark_prompts[
                        : self.num_of_errors_messages_limit
                        - len(current_error_benchmark_prompts)
                    ]
                ]
            )

            if current_num_of_error_prompts > 0:
                self.run_progress_cb.notify_error(
                    BENCHMARKING_TASK_MANAGER_NOTIFY_ERROR_MSG
                )

            logger.info(
                BENCHMARKING_TASK_MANAGER_TASK_INFO_MSG.format(
                    recipe_task=task.recipe_name,
                    connector_id=task.connector.id,
                    progress=task_progress,
                )
            )

        if total_num_of_tasks > 0:
            current_overall_progress = int(current_task_progress / total_num_of_tasks)
        else:
            current_overall_progress = 0

        logger.info(
            BENCHMARKING_TASK_MANAGER_OVERALL_PROGRESS_INFO_MSG.format(
                progress=current_overall_progress
            )
        )

        # Update run progress on the current combined task progress
        if self.run_progress_cb:
            self.run_progress_cb.notify_progress(
                total_num_of_tasks=total_num_of_tasks,
                total_num_of_prompts=await self.total_num_of_prompts.get(),
                completed_num_of_prompts=current_num_of_completed_prompts,
                cancelled_num_of_prompts=current_num_of_cancelled_prompts,
                error_num_of_prompts=current_num_of_error_prompts,
                overall_progress=current_overall_progress,
                cancelled_benchmark_prompts=current_cancelled_benchmark_prompts,
                error_benchmark_prompts=current_error_benchmark_prompts,
            )


# ---------------------------------------------------------------------------------------------------------------------
# Benchmarking Task Class
# ---------------------------------------------------------------------------------------------------------------------
class BenchmarkingTask(BaseModel):
    """
    Represents the arguments and state required for a benchmarking task.

    Attributes:
        cookbook_name (str): The name of the cookbook to be used in the benchmarking task.
        cookbook_instance (Cookbook | None): An instance of the Cookbook class, if available.
        recipe_name (str): The name of the recipe to be used in the benchmarking task.
        recipe_instance (Recipe | None): An instance of the Recipe class, if available.
        prompts_generator (AsyncGenerator[BenchmarkingPrompt, None]): An asynchronous generator for benchmarking prompts
        connector (Connector): The connector used for the benchmarking task.
        start_time (datetime): The start time of the benchmarking task.
        end_time (datetime): The end time of the benchmarking task.
        benchmark_task_manager_cb (BenchmarkingTaskManager): Callback for managing the benchmarking task.
        progress (AtomicInteger): The current progress of the benchmarking task.
        num_of_total_metrics (AtomicInteger): The total number of metrics to be processed.
        num_of_completed_metrics (AtomicInteger): The number of metrics that have been completed.
        num_of_total_prompts (AtomicInteger): The total number of prompts to be processed.
        num_of_completed_prompts (AtomicInteger): The number of prompts that have been completed.
        num_of_error_prompts (AtomicInteger): The number of prompts that resulted in errors.
        num_of_cancelled_prompts (AtomicInteger): The number of prompts that were cancelled.
        error_benchmark_prompts (list[BenchmarkingPrompt]): A list of benchmarking prompts that resulted in errors.
        cancelled_benchmark_prompts (list[BenchmarkingPrompt]): A list of benchmarking prompts that were cancelled.
    """

    class Config:
        arbitrary_types_allowed = True

    cookbook_name: str
    cookbook_instance: Cookbook | None
    recipe_name: str
    recipe_instance: Recipe | None
    prompts_generator: AsyncGenerator[BenchmarkingPrompt, None]
    connector: Connector
    start_time: datetime = datetime.now()
    end_time: datetime = datetime.now()
    benchmark_task_manager_cb: BenchmarkingTaskManager
    progress: AtomicInteger
    num_of_total_metrics: AtomicInteger
    num_of_completed_metrics: AtomicInteger
    num_of_total_prompts: AtomicInteger
    num_of_completed_prompts: AtomicInteger
    num_of_error_prompts: AtomicInteger
    num_of_cancelled_prompts: AtomicInteger
    error_benchmark_prompts: list[BenchmarkingPrompt]
    cancelled_benchmark_prompts: list[BenchmarkingPrompt]

    async def notify_progress(self) -> None:
        """
        Notify the progress of the benchmarking task.

        This method calculates the current progress of the task and updates the benchmarking task manager
        with the latest metrics and prompt counts.
        """
        # Calculate current progress
        await self.progress.set(int(await self.calculate_progress()))

        # Notify the benchmark task manager of the new progress
        await self.benchmark_task_manager_cb.notify_task_progress()

    async def calculate_progress(self) -> float:
        """
        Calculate the progress of the benchmarking task.

        This method calculates the progress based on the number of completed, error, and cancelled prompts
        relative to the total number of prompts. This accounts for 50% of the total progress. The other 50%
        comes from the metrics.
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


# -----------------------------------------------
# Benchmarking Class
# -----------------------------------------------
class Benchmarking:
    sql_create_runner_cache_record = """
        INSERT INTO runner_cache_table(connection_id,recipe_id,dataset_id,prompt_template_id,attack_module_id,
        prompt_index,prompt,target,predicted_results,duration,random_seed,system_prompt)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
    """
    sql_read_runner_cache_record = """
        SELECT * from runner_cache_table WHERE connection_id=? AND recipe_id=? AND prompt_template_id=? AND prompt=?
    """

    async def generate(
        self,
        event_loop: Any,
        runner_args: dict,
        database_instance: DBInterface | None,
        endpoints: list[str],
        run_progress: RunProgress,
        cancel_event: asyncio.Event,
    ) -> ResultArguments | None:
        """
        Asynchronously generates benchmarking results based on the provided runner arguments and
        stores them in the database.

        This method orchestrates the benchmarking process by preparing the environment, running the recipes and
        cookbooks, and collecting the results. It leverages the provided database instance to cache and
        retrieve runner data.

        Args:
            event_loop (Any): The event loop in which asynchronous tasks will be scheduled.
            runner_args (dict): A dictionary containing arguments for the runner.
            database_instance (DBInterface | None): The database interface for storing and retrieving runner data.
            endpoints (list[str]): A list of endpoint identifiers to be used in the benchmarking process.
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
            self.runner_args = runner_args
            self.database_instance = database_instance
            self.endpoints = endpoints
            self.run_progress = run_progress
            self.cancel_event = cancel_event

            # Run benchmarking workflow
            await self.run_workflow()

        except Exception as e:
            run_progress.notify_error(
                BENCHMARKING_GENERATE_ERROR.format(message=str(e))
            )

        finally:
            logger.debug(BENCHMARKING_GENERATE_UPDATE_BENCHMARK_STATUS)
            if cancel_event.is_set():
                run_progress.notify_progress(
                    status=RunStatus.CANCELLED, total_progress=100
                )
            elif run_progress.run_arguments.error_messages:
                run_progress.notify_progress(
                    status=RunStatus.COMPLETED_WITH_ERRORS, total_progress=100
                )
            else:
                run_progress.notify_progress(
                    status=RunStatus.COMPLETED, total_progress=100
                )

    async def run_workflow(self) -> None:
        """
        Orchestrates the benchmarking workflow by loading required instances, generating tasks,
        and processing them asynchronously.

        This method performs the following steps:
        1. Retrieves required arguments from runner_args.
        2. Loads endpoint instances and sets system prompts.
        3. Generates benchmarking tasks.
        4. Creates queues for task processing.
        5. Starts task processing and waits for all tasks to complete.

        Raises:
            RuntimeError: If no endpoints are provided.
        """
        # Retrieve required arguments from runner_args
        self.cookbooks = self.runner_args.get("cookbooks", None)
        self.recipes = self.runner_args.get("recipes", None)
        self.num_of_prompts = self.runner_args.get("num_of_prompts", 0)
        self.random_seed = self.runner_args.get("random_seed", 0)
        self.system_prompt = self.runner_args.get("system_prompt", "")
        self.use_cache = self.runner_args.get("use_cache", True)
        self.error_messages_limit = self.runner_args.get("error_messages_limit", 10)
        self.cancelled_messages_limit = self.runner_args.get(
            "cancelled_messages_limit", 10
        )

        # Load endpoint instances and set system prompts
        if self.endpoints:
            connectors = await self.get_connector_endpoint_instances(self.endpoints)
            logger.debug(BENCHMARKING_RUN_WORKFLOW_LOAD_ENDPOINT_INSTANCES_SUCCESS)
        else:
            raise RuntimeError(
                BENCHMARKING_RUN_WORKFLOW_LOAD_ENDPOINT_INSTANCES_NO_ENDPOINTS_MSG
            )

        if self.system_prompt:
            await self.set_system_prompts(connectors)
            logger.debug(BENCHMARKING_RUN_WORKFLOW_UPDATE_SYSTEM_PROMPT_SUCCESS)
        else:
            logger.warning(BENCHMARKING_RUN_WORKFLOW_UPDATE_SYSTEM_PROMPT_SKIPPED)

        # Create a benchmarking task manager to manage multiple BenchmarkingTask instances and
        # provide feedback to run progress
        benchmark_task_manager = BenchmarkingTaskManager(
            self.run_progress, self.error_messages_limit, self.cancelled_messages_limit
        )

        # Generate benchmarking tasks that will callback to the task manager
        await self.generate_tasks(connectors, benchmark_task_manager)
        logger.info(
            BENCHMARKING_RUN_WORKFLOW_GENERATE_TASKS_INFO.format(
                message=await benchmark_task_manager.total_num_of_tasks.get(),
                total_prompts=await benchmark_task_manager.total_num_of_prompts.get(),
            )
        )

        # Notify the benchmark task manager to update task progress
        await benchmark_task_manager.notify_task_progress()

        # Check if there are benchmark tasks or prompts for processing
        if (
            not benchmark_task_manager.tasks
            or await benchmark_task_manager.total_num_of_prompts.get() == 0
        ):
            return

        # Create queues for task processing
        completed_query_llm_queue = asyncio.Queue()  # Queue for external LLM completion
        metrics_calc_queue = asyncio.Queue()  # Queue for metrics calculation

        # Start task processing
        asyncio.create_task(
            self.completed_query_llm_handler(
                completed_query_llm_queue, metrics_calc_queue
            )
        )
        asyncio.create_task(self.metrics_calc_handler(metrics_calc_queue))

        # Wait for all tasks to complete
        logger.info(
            BENCHMARKING_RUN_WORKFLOW_QUERY_LLM_INFO.format(
                num_of_tasks=await benchmark_task_manager.total_num_of_tasks.get()
            )
        )
        tasks = [
            self.query_llm(task, completed_query_llm_queue)
            for task in benchmark_task_manager.tasks
        ]
        await asyncio.gather(*tasks)

        # Signal completion of task processing
        await completed_query_llm_queue.put(None)

        # Wait until all tasks in the queues are processed
        await completed_query_llm_queue.join()
        await metrics_calc_queue.join()

    async def query_llm(
        self, task: BenchmarkingTask, completed_query_llm_queue: asyncio.Queue
    ) -> None:
        """
        Process each prompt in the given benchmarking task by querying the LLM and handling caching.

        This method iterates over the prompts in the task, checks for cached results, and if not found,
        queries the LLM for predictions. It then caches the results and updates the task with completed,
        error, and cancelled prompts.

        Args:
            task (BenchmarkingTask): The benchmarking task containing prompts to be processed.
            completed_query_llm_queue (asyncio.Queue): The queue to put the completed task into after processing.
        """
        # Log the task start and end time
        task.start_time = datetime.now()
        task.end_time = datetime.now()

        async def query_benchmark_prompt(benchmark_prompt: BenchmarkingPrompt):
            """
            Process a single benchmarking prompt by querying the LLM and handling caching.

            This method checks for a cancellation event, retrieves cached results if available,
            queries the LLM for predictions if not cached, and updates the task with completed,
            error, and cancelled prompts. It also notifies the task progress.

            Args:
                benchmark_prompt (BenchmarkingPrompt): The benchmarking prompt to be processed.
            """
            if self.cancel_event.is_set():
                logger.warning(BENCHMARKING_QUERY_LLM_CANCEL_SET_WARNING)
                await task.num_of_cancelled_prompts.increment()
                task.cancelled_benchmark_prompts.append(benchmark_prompt)

                await task.notify_progress()  # Recalculate progress
                return

            cache_record = None
            if self.use_cache:
                try:
                    cache_record = Storage.read_database_record(
                        self.database_instance,
                        (
                            benchmark_prompt.conn_id,
                            benchmark_prompt.rec_id,
                            benchmark_prompt.pt_id,
                            benchmark_prompt.connector_prompt.prompt,
                        ),
                        Benchmarking.sql_read_runner_cache_record,
                    )
                except Exception:
                    cache_record = None

            # If cache record does not exist, perform prediction and cache the result
            if cache_record is None:
                try:
                    benchmark_prompt.connector_prompt = await Connector.get_prediction(
                        benchmark_prompt.connector_prompt, task.connector
                    )
                    Storage.create_database_record(
                        self.database_instance,
                        benchmark_prompt.to_tuple(),
                        Benchmarking.sql_create_runner_cache_record,
                    )
                    await task.num_of_completed_prompts.increment()
                except Exception:
                    await task.num_of_error_prompts.increment()
                    task.error_benchmark_prompts.append(benchmark_prompt)
            else:
                # Load result from cache
                benchmark_prompt = BenchmarkingPrompt.from_tuple(cache_record)
                await task.num_of_completed_prompts.increment()

            await task.notify_progress()  # Recalculate progress

        # Create a list of tasks for processing prompts
        tasks = [
            asyncio.create_task(query_benchmark_prompt(benchmark_prompt))
            async for benchmark_prompt in task.prompts_generator
        ]

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        # Task completed
        task.end_time = datetime.now()
        time_taken = (task.end_time - task.start_time).total_seconds()
        logger.info(
            BENCHMARKING_QUERY_LLM_TASK_DONE.format(
                recipe_name=task.recipe_name,
                connector_id=task.connector.id,
                num_of_prompts=await task.num_of_total_prompts.get(),
                time_taken=time_taken,
                num_completed_benchmark_prompts=await task.num_of_completed_prompts.get(),
                num_cancelled_benchmark_prompts=await task.num_of_cancelled_prompts.get(),
                num_error_benchmark_prompts=await task.num_of_error_prompts.get(),
            )
        )

        # # Put the task in the completed_query_llm_queue after all are complete
        # await completed_query_llm_queue.put(task)

    async def completed_query_llm_handler(
        self,
        completed_query_llm_queue: asyncio.Queue[BenchmarkingTask],
        metrics_calc_queue: asyncio.Queue[BenchmarkingTask],
    ) -> None:
        """
        Continuously handles tasks from the completed query LLM queue and adds them to the metrics calculation queue.

        This method runs an infinite loop to check for tasks in the completed_query_llm_queue.
        If the cancel_event is set, it logs a warning and breaks the loop.

        Otherwise, it retrieves each task from the completed_query_llm_queue and places it into the metrics_calc_queue
        for further processing.

        Args:
            completed_query_llm_queue (asyncio.Queue[BenchmarkingTask]): The queue from which completed tasks
                                                                         are retrieved for LLM querying.
            metrics_calc_queue (asyncio.Queue[BenchmarkingTask]): The queue to which tasks are added for
                                                                  metrics calculation.
        """
        while True:
            if self.cancel_event.is_set():
                logger.warning(BENCHMARKING_QUERY_LLM_HANDLER_CANCEL_SET_WARNING)
                break

            task = await completed_query_llm_queue.get()
            await metrics_calc_queue.put(task)

            # Check if it's the sentinel, which signals the end
            if task is None:
                completed_query_llm_queue.task_done()  # Mark the task as done before breaking
                break

            completed_query_llm_queue.task_done()

    async def metrics_calc_handler(self, metrics_calc_queue: asyncio.Queue) -> None:
        """
        Continuously processes tasks from the metrics calculation queue.

        This method runs an infinite loop to check for tasks in the metrics_calc_queue. If the cancel_event is set,
        it logs a warning and breaks the loop. Otherwise, it retrieves each task from the metrics_calc_queue and
        processes it.

        Args:
            metrics_calc_queue (asyncio.Queue): The queue from which tasks are retrieved for metrics calculation.
        """
        while True:
            if self.cancel_event.is_set():
                logger.warning(BENCHMARKING_METRICS_CALC_HANDLER_CANCEL_SET_WARNING)
                break

            task = await metrics_calc_queue.get()
            # Check if it's the sentinel, which signals the end
            if task is None:
                metrics_calc_queue.task_done()
                break

            # Process the task
            await self.generate_metrics_results(task)
            metrics_calc_queue.task_done()

    async def generate_metrics_results(self, task: BenchmarkingTask) -> None:
        print("Generate Metrics")
        # print(task.completed_benchmark_prompts)

    async def generate_prompts(
        self, recipe_instance: Recipe, connector: Connector
    ) -> tuple[int, AsyncGenerator[BenchmarkingPrompt, None]]:
        """
        Asynchronously generates prompts based on the provided recipe instance and connector.

        This method retrieves prompt templates from storage, counts the number of prompts that will be generated,
        and returns both the count and an asynchronous generator for the prompts.

        Args:
            recipe_instance (Recipe): The recipe instance containing datasets and prompt templates.
            connector (Connector): The connector instance used for generating prompts.

        Returns:
            tuple[int, AsyncGenerator[BenchmarkingPrompt, None]]: A tuple containing the count of prompts and an
            asynchronous generator for the prompts.
        """
        pt_id = "no-template"
        templates = {}
        if recipe_instance.prompt_templates:
            for pt_id in recipe_instance.prompt_templates:
                # Retrieve the prompt template information from storage as a generator
                pt_info_gen = Storage.read_object_with_iterator(
                    EnvVariables.PROMPT_TEMPLATES.name,
                    pt_id,
                    "json",
                    iterator_keys=["template"],
                )
                # Get the first item from the generator, which contains the template data
                pt_info = next(pt_info_gen["template"])
                # Create a Jinja2 template from the retrieved template data
                templates[pt_id] = Template(pt_info)

        # Count the number of prompts that will be generated
        count = 0
        async for _ in self.prompt_generator(recipe_instance, templates, connector):
            count += 1

        # Return the count and the generator
        return count, self.prompt_generator(recipe_instance, templates, connector)

    async def prompt_generator(
        self, recipe_instance: Recipe, templates: dict, connector: Connector
    ) -> AsyncGenerator[BenchmarkingPrompt, None]:
        """
        Asynchronously generates prompts based on the provided recipe instance and templates.

        This method iterates over datasets and templates to render prompts and yield them.
        If no templates are available, it yields the modified prompts from the datasets directly.

        Args:
            recipe_instance (Recipe): The recipe instance containing datasets and prompt templates.
            templates (dict): A dictionary of prompt templates.
            connector (Connector): The connector instance used for generating prompts.

        Yields:
            BenchmarkingPrompt: An instance of BenchmarkingPrompt containing all the necessary
            information for processing the prompt.
        """
        for ds_id in recipe_instance.datasets:
            async for prompt_index, prompt in self.get_dataset_prompts(ds_id):
                modified_prompts = [("", prompt["input"])]

                # If templates are available, render the modified prompts using the templates
                if templates:
                    for pt_id, jinja2_template in templates.items():
                        # Render the modified prompt using the Jinja2 template
                        for (
                            modified_attack_module_id,
                            modified_prompt,
                        ) in modified_prompts:
                            rendered_prompt = jinja2_template.render(
                                {"prompt": modified_prompt}
                            )
                            prompt_args = await self.yield_benchmarking_prompt(
                                connector.id,
                                recipe_instance.id,
                                pt_id,
                                ds_id,
                                modified_attack_module_id,
                                prompt_index,
                                rendered_prompt,
                                prompt["target"],
                            )
                            yield prompt_args
                # If no templates are available, yield the modified prompts directly
                else:
                    for modified_attack_module_id, modified_prompt in modified_prompts:
                        prompt_args = await self.yield_benchmarking_prompt(
                            connector.id,
                            recipe_instance.id,
                            pt_id,
                            ds_id,
                            modified_attack_module_id,
                            prompt_index,
                            modified_prompt,
                            prompt["target"],
                        )
                        yield prompt_args

    async def yield_benchmarking_prompt(
        self,
        conn_id: str,
        rec_id: str,
        pt_id: str,
        ds_id: str,
        attack_module_id: str,
        prompt_index: int,
        prompt_text: str,
        target: str,
    ) -> BenchmarkingPrompt:
        """
        Asynchronously prepares the arguments required for a benchmarking prompt.

        This method takes various identifiers and prompt information, and prepares a
        BenchmarkingPrompt object which contains all the necessary details for processing
        the prompt in the benchmarking workflow.

        Args:
            conn_id (str): The ID of the connection.
            rec_id (str): The ID of the recipe instance.
            pt_id (str): The ID of the prompt template.
            ds_id (str): The ID of the dataset.
            attack_module_id (str): The ID of the attack module.
            prompt_index (int): The index of the prompt in the dataset.
            prompt_text (str): The text of the prompt.
            target (str): The target for the prompt.

        Returns:
            BenchmarkingPrompt: An instance of BenchmarkingPrompt containing all the necessary
            information for processing the prompt.
        """
        return BenchmarkingPrompt(
            conn_id=conn_id,
            rec_id=rec_id,
            ds_id=ds_id,
            pt_id=pt_id,
            random_seed=self.random_seed,
            system_prompt=self.system_prompt,
            attack_module_id=attack_module_id,
            connector_prompt=ConnectorPromptArguments(
                prompt_index=prompt_index,
                prompt=prompt_text,
                target=target,
            ),
        )

    async def get_dataset_prompts(
        self, ds_id: str
    ) -> AsyncGenerator[tuple[int, dict], None]:
        """
        Asynchronously retrieves prompts from a dataset based on the specified dataset ID.

        This method determines the total number of prompts in the dataset and generates a list of prompt indices.
        If a specific number of prompts is requested (num_of_prompts), it will randomly select that many prompts
        using the provided random seed. Otherwise, it will retrieve all prompts. Each prompt is then fetched and
        yielded along with its index.

        Args:
            ds_id (str): The ID of the dataset from which to retrieve prompts.

        Yields:
            tuple[int, dict]: A tuple containing the index of the prompt and the prompt data itself.

        Raises:
            ValueError: If the dataset ID is invalid or the dataset cannot be read.
        """
        # Get dataset arguments
        ds_args = Dataset.read(ds_id)

        # Generate a list of prompt indices based on num_of_prompts and random_seed
        if (
            self.num_of_prompts == 0
            or self.num_of_prompts > ds_args.num_of_dataset_prompts
        ):
            prompt_indices = range(ds_args.num_of_dataset_prompts)
        else:
            random.seed(self.random_seed)
            prompt_indices = random.sample(
                range(ds_args.num_of_dataset_prompts), self.num_of_prompts
            )

        # Use for loop to iterate over the dataset examples
        prompts_gen_index = 0
        for prompts_data in ds_args.examples:
            if prompts_gen_index in prompt_indices:
                yield prompts_gen_index, prompts_data
            prompts_gen_index += 1

    async def generate_tasks(
        self,
        connectors: list[Connector],
        benchmark_task_manager: BenchmarkingTaskManager,
    ) -> None:
        """
        Asynchronously generates benchmarking tasks based on provided cookbooks and recipes.

        This method creates instances of `BenchmarkingTask` for each recipe found in the provided cookbooks
        and standalone recipes. Each task is then registered with the benchmark task manager.

        Args:
            connectors (list[Connector]): A list of Connector instances to be used for each task.
            benchmark_task_manager (BenchmarkingTaskManager): The task manager to register new tasks with.

        Raises:
            RuntimeError: If there is an error loading a cookbook or recipe.
        """
        if self.cookbooks:
            for cookbook in self.cookbooks:
                try:
                    cookbook_instance = Cookbook.load(cookbook)
                    for recipe in cookbook_instance.recipes:
                        recipe_instance = Recipe.load(recipe)

                        # Loop through each connector and create a task for each
                        for connector in connectors:
                            prompts, prompts_gen = await self.generate_prompts(
                                recipe_instance, connector
                            )
                            new_task = BenchmarkingTask(
                                cookbook_name=cookbook,
                                cookbook_instance=cookbook_instance,
                                recipe_name=recipe,
                                recipe_instance=recipe_instance,
                                prompts_generator=prompts_gen,
                                connector=connector,
                                benchmark_task_manager_cb=benchmark_task_manager,
                                progress=AtomicInteger(0),
                                num_of_total_metrics=AtomicInteger(0),
                                num_of_completed_metrics=AtomicInteger(0),
                                num_of_total_prompts=AtomicInteger(0),
                                num_of_completed_prompts=AtomicInteger(0),
                                num_of_error_prompts=AtomicInteger(0),
                                num_of_cancelled_prompts=AtomicInteger(0),
                                error_benchmark_prompts=[],
                                cancelled_benchmark_prompts=[],
                            )
                            await new_task.num_of_total_metrics.set(
                                len(recipe_instance.metrics)
                            )
                            await new_task.num_of_total_prompts.set(prompts)
                            await benchmark_task_manager.register_new_task(new_task)
                except Exception as e:
                    raise RuntimeError(
                        BENCHMARKING_GENERATE_TASKS_COOKBOOK_ERROR_MSG.format(
                            cookbook_name=cookbook, message=str(e)
                        )
                    )

        if self.recipes:
            for recipe in self.recipes:
                try:
                    recipe_instance = Recipe.load(recipe)

                    # Loop through each connector and create a task for each
                    for connector in connectors:
                        prompts, prompts_gen = await self.generate_prompts(
                            recipe_instance, connector
                        )
                        new_task = BenchmarkingTask(
                            cookbook_name="",  # No cookbook associated
                            cookbook_instance=None,  # No cookbook instance
                            recipe_name=recipe,
                            recipe_instance=recipe_instance,
                            prompts_generator=prompts_gen,
                            connector=connector,
                            benchmark_task_manager_cb=benchmark_task_manager,
                            progress=AtomicInteger(0),
                            num_of_total_metrics=AtomicInteger(0),
                            num_of_completed_metrics=AtomicInteger(0),
                            num_of_total_prompts=AtomicInteger(0),
                            num_of_completed_prompts=AtomicInteger(0),
                            num_of_error_prompts=AtomicInteger(0),
                            num_of_cancelled_prompts=AtomicInteger(0),
                            error_benchmark_prompts=[],
                            cancelled_benchmark_prompts=[],
                        )
                        await new_task.num_of_total_metrics.set(
                            len(recipe_instance.metrics)
                        )
                        await new_task.num_of_total_prompts.set(prompts)
                        await benchmark_task_manager.register_new_task(new_task)
                except Exception as e:
                    raise RuntimeError(
                        BENCHMARKING_GENERATE_TASKS_RECIPE_ERROR_MSG.format(
                            recipe_name=recipe, message=str(e)
                        )
                    )

    async def get_connector_endpoint_instances(
        self, endpoints: list[str]
    ) -> list[Connector]:
        """
        Asynchronously loads connector endpoint instances from a list of endpoint names.

        Args:
            endpoints (list[str]): A list of endpoint names to be loaded.

        Returns:
            list[Connector]: A list of Connector instances created from the provided endpoints.

        Raises:
            RuntimeError: If any endpoint fails to load, an error message is logged and the exception is raised.
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

    async def set_system_prompts(self, connectors: list[Connector]) -> None:
        """
        Asynchronously sets the system prompt for each connector in the provided list.

        Args:
            connectors (list[Connector]): A list of Connector instances to update.

        Raises:
            RuntimeError: If setting the system prompt fails for any connector,
                          an error message is logged and the exception is raised.
        """
        for connector in connectors:
            try:
                connector.set_system_prompt(self.system_prompt)
            except Exception as e:
                raise RuntimeError(
                    BENCHMARKING_SET_SYSTEM_PROMPTS_FAILED_MSG.format(
                        connector_name=connector.id, message=str(e)
                    )
                )
