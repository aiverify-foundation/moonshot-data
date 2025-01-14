from __future__ import annotations

import asyncio

from moonshot.src.benchmark.task.task_generator import TaskGenerator
from moonshot.src.connectors.connector import Connector
from moonshot.src.connectors_endpoints.connector_endpoint import ConnectorEndpoint
from moonshot.src.results.result_arguments import ResultArguments
from moonshot.src.runs.run_progress import RunProgress
from moonshot.src.runs.run_status import RunStatus
from moonshot.src.storage.db_interface import DBInterface
from moonshot.src.utils.log import configure_logger

# Create a logger for this module
logger = configure_logger(__name__)

# ---------------------------------------------------------------------------------------------------------------------
# Benchmarking Messages
# ---------------------------------------------------------------------------------------------------------------------
# Cancellation Messages
BENCHMARKING_CANCEL_SET_WARNING_MSG = (
    "[Benchmarking] Cancellation detected. Stopping now..."
)

# Error Messages
BENCHMARKING_GENERATE_CANCELLED_ERROR_MSG = (
    "[Benchmarking] Benchmarking cancelled because of an error: {message}"
)
BENCHMARKING_GENERATE_EXCEPTION_ERROR_MSG = (
    "[Benchmarking] Benchmarking encountered an error: {message}"
)
BENCHMARKING_GENERATE_ERROR_MSG = (
    "[Benchmarking] Error during result generation: {message}"
)
BENCHMARKING_GENERATE_NO_DATABASE_INSTANCE_MSG = (
    "[Benchmarking] No database found. Stopping the process."
)
BENCHMARKING_GENERATE_TASKS_COOKBOOK_ERROR_MSG = "[Benchmarking] Couldn't load cookbook '{cookbook_name}': {message}. Stopping the process."  # noqa: E501
BENCHMARKING_GENERATE_TASKS_RECIPE_ERROR_MSG = "[Benchmarking] Couldn't load recipe '{recipe_name}': {message}. Stopping the process."  # noqa: E501
BENCHMARKING_GET_CONNECTOR_ENDPOINT_INSTANCES_LOAD_FAILED_MSG = "[Benchmarking] Couldn't load endpoint '{endpoint_name}': {message}. Stopping the process."  # noqa: E501
BENCHMARKING_SET_SYSTEM_PROMPTS_FAILED_MSG = "[Benchmarking] Couldn't set system prompt on '{connector_name}': {message}. Exiting."  # noqa: E501
BENCHMARKING_SET_STATUS_DEBUG_MSG = (
    "[Benchmarking] Benchmarking status changed to {new_status}."
)


# Status Update Messages
BENCHMARKING_GENERATE_UPDATE_BENCHMARK_STATUS_MSG = (
    "[Benchmarking] Updating benchmark status."
)

# Run Messages
BENCHMARKING_RUN_CHECK_TASKS_PROMPTS_MSG = (
    "[Benchmarking] No tasks or prompts found. Exiting now."
)
BENCHMARKING_RUN_DEBUG_MSG = "[Benchmarking] Running with: run_id={run_id}, runner_id={runner_id}, runner_args={runner_args}, database_instance={database_instance}, endpoints={endpoints}, run_progress={run_progress}, cancel_event={cancel_event}"  # noqa: E501
BENCHMARKING_RUN_GENERATE_TASKS_INFO_MSG = (
    "[Benchmarking] Created {message} task(s) with {total_prompts} prompts."
)
BENCHMARKING_RUN_LOAD_ENDPOINT_INSTANCES_NO_ENDPOINTS_MSG = (
    "[Benchmarking] No endpoints specified. Exiting."
)
BENCHMARKING_RUN_LOAD_ENDPOINT_INSTANCES_SUCCESS_MSG = (
    "[Benchmarking] Endpoints loaded successfully."
)
BENCHMARKING_RUN_QUERY_LLM_INFO_MSG = "[Benchmarking] Starting LLM queries for {num_of_tasks} task(s) with {num_of_benchmark_prompts} benchmark prompt(s)."  # noqa: E501
BENCHMARKING_RUN_UPDATE_SYSTEM_PROMPT_SKIPPED_MSG = (
    "[Benchmarking] Skipped updating system prompts."
)
BENCHMARKING_RUN_UPDATE_SYSTEM_PROMPT_SUCCESS_MSG = (
    "[Benchmarking] System prompts updated successfully."
)


# -----------------------------------------------
# Benchmarking Class
# -----------------------------------------------
class Benchmarking:
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
        Generate benchmarking tasks and handle the benchmarking process.

        Args:
            event_loop (asyncio.AbstractEventLoop): The event loop to run asynchronous tasks.
            run_id (str): A unique identifier for this run.
            runner_id (str): A unique identifier for the runner.
            runner_args (dict): A dictionary containing various settings for the runner.
            database_instance (DBInterface | None): The database instance to store results, or None if not used.
            endpoints (list[str]): A list of endpoint identifiers.
            run_progress (RunProgress): An object to track and update the progress of the run.
            cancel_event (asyncio.Event): An event to signal if the benchmarking process should be canceled.

        Returns:
            ResultArguments | None: The result arguments if the process completes successfully, otherwise None.
        """
        try:
            # Initialize parameters for the run
            self._initialize_parameters(
                cancel_event,
                database_instance,
                endpoints,
                run_id,
                run_progress,
                runner_args,
                runner_id,
            )

            # Set the status to RUNNING_BENCHMARK
            self.set_status(RunStatus.RUNNING_BENCHMARK)

            # Check if the process is cancelled before starting
            if self.cancel_event.is_set():
                self.set_status(RunStatus.CANCELLED)
                return

            # Log the assignment of parameters using the debug message
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

            # Load connectors and set prompts
            await self._load_connectors_and_set_prompts()

            # Check if the process is cancelled after loading connectors and setting prompts
            if self.cancel_event.is_set():
                self.set_status(RunStatus.CANCELLED)
                return

            # Check if the database instance is not provided
            if not self.database_instance:
                self._handle_no_database_instance()
            else:
                task_generator = TaskGenerator(
                    self.cancel_event,
                    self.concurrency_limit,
                    self.connectors,
                    self.cookbooks or [],
                    self.database_instance,
                    self.prompt_selection_percentage,
                    self.random_seed,
                    self.recipes or [],
                    self.handle_task_progress,
                    self.use_cache,
                )
                is_success, response = await task_generator.create_tasks()
                if is_success:
                    self.benchmark_tasks = response.get("tasks", [])
                    self.benchmark_prompts = response.get("tasks_prompts", 0)
                    self.benchmark_updates = response.get("tasks_updates", {})

                    # Start running benchmarking
                    await self.run()

                else:
                    # No tasks, prompts, or metrics to run
                    pass

        except asyncio.CancelledError as e:
            # If the task was explicitly cancelled by higher-level logic
            logger.warning(
                BENCHMARKING_GENERATE_CANCELLED_ERROR_MSG.format(message=str(e))
            )

        except Exception as e:
            # Log the error and set the status to COMPLETED_WITH_ERRORS
            logger.error(
                BENCHMARKING_GENERATE_EXCEPTION_ERROR_MSG.format(message=str(e))
            )

        finally:
            self._finalize_progress()

        # ------------------------------------------------------------------------------
        # Prepare ResultArguments
        # ------------------------------------------------------------------------------
        # try:
        #     result_args = ResultArguments(
        #         # Mandatory values
        #         id=self.run_progress.run_arguments.runner_id,
        #         start_time=self.run_progress.run_arguments.start_time,
        #         end_time=self.run_progress.run_arguments.end_time,
        #         duration=self.run_progress.run_arguments.duration,
        #         status=self.run_progress.run_arguments.status,
        #         raw_results=self.run_progress.run_arguments.raw_results,
        #         params={
        #             "recipes": self.recipes,
        #             "cookbooks": self.cookbooks,
        #             "endpoints": self.endpoints,
        #             "num_of_prompts": self.num_of_prompts,
        #             "random_seed": self.random_seed,
        #             "system_prompt": self.system_prompt,
        #         },
        #     )

        # except Exception as e:
        #     self.run_progress.notify_error(
        #         LOG_BENCHMARKING_GENERATE_FAILED_PREPARE_RESULTS.format(str(e))
        #     )

        # finally:
        #     logger.info(
        #         LOG_BENCHMARKING_GENERATE_PREPARING_RESULTS_DURATION.format(
        #             time.perf_counter() - start_time
        #         )
        #     )
        #     return result_args

    def set_status(self, new_status: RunStatus) -> None:
        """
        Update the status of the benchmarking run and perform real-time updates if a progress function is provided.

        Args:
            new_status (RunStatus): The new status to set for the benchmarking run.
        """
        logger.debug(
            BENCHMARKING_SET_STATUS_DEBUG_MSG.format(new_status=new_status.value)
        )

        # If a progress function is provided, perform real-time updates
        if self.run_progress:
            self.run_progress.notify_progress(status=new_status)

    def _initialize_parameters(
        self,
        cancel_event: asyncio.Event,
        database_instance: DBInterface | None,
        endpoints: list[str],
        run_id: str,
        run_progress: RunProgress,
        runner_args: dict,
        runner_id: str,
    ) -> None:
        """
        Initialize the parameters required to start the benchmarking process.

        Args:
            cancel_event (asyncio.Event): An event to signal if the benchmarking process should be canceled.
            database_instance (DBInterface | None): The database instance to store results, or None if not used.
            endpoints (list[str]): A list of endpoint identifiers.
            run_id (str): A unique identifier for this run.
            run_progress (RunProgress): An object to track and update the progress of the run.
            runner_args (dict): A dictionary containing various settings for the runner.
            runner_id (str): A unique identifier for the runner.
        """
        self.cancel_event = cancel_event
        self.database_instance = database_instance
        self.endpoints = endpoints
        self.run_id = run_id
        self.run_progress = run_progress
        self.runner_args = runner_args
        self.runner_id = runner_id

        # Extract necessary settings from runner_args
        self.concurrency_limit: int = runner_args.get("concurrency_limit", 10)
        self.cookbooks: list[str] | None = runner_args.get("cookbooks", None)
        self.prompt_selection_percentage: int = runner_args.get(
            "prompt_selection_percentage", 100
        )
        self.random_seed: int = runner_args.get("random_seed", 0)
        self.recipes: list[str] | None = runner_args.get("recipes", None)
        self.system_prompt: str = runner_args.get("system_prompt", "")
        self.use_cache: bool = runner_args.get("use_cache", True)

    def _handle_no_database_instance(self) -> None:
        """
        Handle the case where there is no database instance.

        This method will inform the progress tracker about the error and raise a RuntimeError
        if there is no database instance to store the results.

        Raises:
            RuntimeError: Raised when no database instance is provided. An error message is logged
            and a RuntimeError is raised.
        """
        # Inform the progress tracker about the error
        self.run_progress.notify_error(BENCHMARKING_GENERATE_NO_DATABASE_INSTANCE_MSG)

        # Raise a RuntimeError because there is no database instance
        raise RuntimeError(BENCHMARKING_GENERATE_NO_DATABASE_INSTANCE_MSG)

    async def _load_connectors_and_set_prompts(self) -> None:
        """
        Asynchronously load connectors and set system prompts.

        This method performs the following steps:
        1. Loads the connectors using the provided endpoints.
        2. Sets the system prompts for each connector.

        Raises:
            RuntimeError: If no endpoints are provided or if setting the system prompt fails for any connector.
        """
        if self.endpoints:
            self.connectors = await self.get_connector_endpoint_instances(
                self.endpoints
            )
            logger.debug(BENCHMARKING_RUN_LOAD_ENDPOINT_INSTANCES_SUCCESS_MSG)
        else:
            raise RuntimeError(
                BENCHMARKING_RUN_LOAD_ENDPOINT_INSTANCES_NO_ENDPOINTS_MSG
            )

        if self.system_prompt:
            await self.set_system_prompts()
            logger.debug(BENCHMARKING_RUN_UPDATE_SYSTEM_PROMPT_SUCCESS_MSG)
        else:
            logger.warning(BENCHMARKING_RUN_UPDATE_SYSTEM_PROMPT_SKIPPED_MSG)

    def _finalize_progress(self) -> None:
        """
        Finish the benchmarking process and update the run status.

        This method checks if the cancel event has been triggered or if there are any error messages,
        and updates the run progress status based on these checks.
        """
        logger.debug(BENCHMARKING_GENERATE_UPDATE_BENCHMARK_STATUS_MSG)
        if self.cancel_event.is_set():
            self.run_progress.notify_progress(
                status=RunStatus.CANCELLED, current_progress=100
            )
        elif self.run_progress.run_arguments.error_messages:
            self.run_progress.notify_progress(
                status=RunStatus.COMPLETED_WITH_ERRORS,
                current_progress=100,
            )
        else:
            self.run_progress.notify_progress(
                status=RunStatus.COMPLETED_BENCHMARK, current_progress=100
            )

    async def run(self) -> None:
        """
        Execute the benchmarking tasks asynchronously.

        This method logs the start of the benchmarking process, creates asyncio tasks for each
        benchmarking task, and gathers the results while handling any exceptions.

        Raises:
            Any exceptions raised during the execution of the asyncio tasks.
        """
        # Log the start of the benchmarking process with the number of tasks and prompts
        logger.info(
            BENCHMARKING_RUN_QUERY_LLM_INFO_MSG.format(
                num_of_tasks=len(self.benchmark_tasks),
                num_of_benchmark_prompts=self.benchmark_prompts,
            )
        )

        # Create asyncio tasks for each benchmarking task
        asyncio_tasks = [
            asyncio.create_task(task.process_task()) for task in self.benchmark_tasks
        ]

        # Gather asyncio tasks and handle exceptions
        await asyncio.gather(*asyncio_tasks, return_exceptions=True)

    async def handle_task_progress(self, task_progress: dict) -> None:
        """
        Asynchronously handle and update the progress of a specific benchmarking task.

        This method prints the task progress for debugging purposes.

        Args:
            task_progress (dict): A dictionary containing the progress update for the task.
        """
        print("*" * 10, "Benchmarking::handle_task_progress", "*" * 10)
        print(task_progress)

    #     if task_uuid and task_update:
    #         # Update the benchmark updates with the new task update
    #         self.benchmark_updates.update({task_uuid: task_update})

    #         # Create a structure to aggregate the progress results
    #         aggregated_result = create_run_progress_structure()

    #         # Aggregate progress data from all tasks
    #         for task in self.benchmark_updates.values():
    #             aggregated_result["cancelled_prompts"].extend(task["cancelled_prompts"])
    #             aggregated_result["completed_prompts"].extend(task["completed_prompts"])
    #             aggregated_result["current_error_messages"].extend(
    #                 task["current_error_messages"]
    #             )
    #             aggregated_result["error_prompts"].extend(task["error_prompts"])
    #             aggregated_result["num_of_prompts_cancelled"] += task[
    #                 "num_of_prompts_cancelled"
    #             ]
    #             aggregated_result["num_of_prompts_completed"] += task[
    #                 "num_of_prompts_completed"
    #             ]
    #             aggregated_result["num_of_prompts_error"] += task["num_of_prompts_error"]
    #             aggregated_result["num_of_prompts_pending"] += task[
    #                 "num_of_prompts_pending"
    #             ]
    #             aggregated_result["num_of_prompts_running_metrics_calculation"] += task[
    #                 "num_of_prompts_running_metrics_calculation"
    #             ]
    #             aggregated_result["num_of_prompts_running_query"] += task[
    #                 "num_of_prompts_running_query"
    #             ]
    #             aggregated_result["num_of_prompts_total"] += task["num_of_prompts_total"]
    #             aggregated_result["num_of_tasks_total"] += 1
    #             aggregated_result["pending_prompts"].extend(task["pending_prompts"])
    #             aggregated_result["running_prompts"].extend(task["running_prompts"])

    #         # Calculate the current progress percentage
    #         if self.benchmark_tasks:
    #             aggregated_result["current_progress"] = int(
    #                 aggregated_result["num_of_prompts_completed"]
    #                 / aggregated_result["num_of_prompts_total"]
    #                 * 100
    #             )

    #         if aggregated_result["current_error_messages"]:
    #             # Update the run status to RUNNING_BENCHMARK_WITH_ERRORS
    #             self.run_progress.notify_progress(status=RunStatus.RUNNING_BENCHMARK_WITH_ERRORS)

    #         # Notify the run progress with the aggregated results
    #         self.run_progress.notify_progress(**aggregated_result)

    async def get_connector_endpoint_instances(
        self, endpoints: list[str]
    ) -> list[Connector]:
        """
        Asynchronously load connector endpoint instances from a list of endpoint identifiers.

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
                # Create a Connector instance from the endpoint and add it to the connectors list
                connectors.append(Connector.create(ConnectorEndpoint.read(endpoint)))

            except Exception as e:
                # Log and raise an error if the endpoint fails to load
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
            RuntimeError: If setting the system prompt fails for any connector, an error message is logged
                          and the exception is raised.
        """
        for connector in self.connectors:
            try:
                # Attempt to set the system prompt for the current connector
                connector.set_system_prompt(self.system_prompt)

            except Exception as e:
                # Log and raise an error if setting the system prompt fails for the current connector
                raise RuntimeError(
                    BENCHMARKING_SET_SYSTEM_PROMPTS_FAILED_MSG.format(
                        connector_name=connector.id, message=str(e)
                    )
                )
