import argparse

from cmd.overrides import apply_llm_overrides, apply_rag_overrides
from src.clients import OllamaLLM
from src.config import get_llm_settings, get_rag_settings
from src.rag import AnswerGenerationStage, LLMVerifier


def run(args: argparse.Namespace) -> None:
    """
    Run stage two: generate answers from a chunks JSONL into a submission JSONL.

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
    verifier = LLMVerifier(llm=llm, prompt_name=rag.verifier_prompt_name)
    stage = AnswerGenerationStage(llm=llm, settings=rag, verifier=verifier)
    stage.run(args.input, args.output, batch_size=args.batch_size)
