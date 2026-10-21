"""Task-type registry. Add a new task type by subclassing tasks.base.Task and registering it here."""
from .generate import GenerateTask
from .ifeval import IFEvalTask
from .judge import JudgeTask
from .mcq import MCQTask

TASK_TYPES = {"mcq": MCQTask, "generate": GenerateTask, "ifeval": IFEvalTask, "judge": JudgeTask}
