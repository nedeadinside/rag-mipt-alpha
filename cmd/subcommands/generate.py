import argparse

from hack_optimization import GenerateStage

from cmd.overrides import apply_llm_overrides, apply_rag_overrides
from src.clients import OllamaLLM
from src.config import get_llm_settings, get_rag_settings


def run(args: argparse.Namespace) -> None:
    """
    Run the generate stage: answer verified questions into a submission JSONL.

    :param args: Parsed CLI namespace.
    """
    llm_cfg = apply_llm_overrides(args, get_llm_settings())
    rag = apply_rag_overrides(args, get_rag_settings())

    llm = OllamaLLM(
        llm_cfg.model_name,
        llm_cfg.base_url,
        llm_cfg.temperature,
        top_p=llm_cfg.top_p,
        top_k=llm_cfg.top_k,
    )
    stage = GenerateStage(
        llm=llm,
        prompt_name=rag.prompt_name,
        refusal_text_name=rag.refusal_text_name,
    )
    stage.run(args.input, args.output, batch_size=args.batch_size)
