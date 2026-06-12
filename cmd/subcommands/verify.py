import argparse

from hack_optimization import VerifyStage

from src.clients import OllamaLLM
from src.rag import LLMVerifier


def run(args: argparse.Namespace) -> None:
    """
    Run the verify stage: annotate chunks with a relevance verdict into a JSONL.

    :param args: Parsed CLI namespace.
    """
    llm = OllamaLLM(
        args.llm_model_name,
        args.llm_base_url,
        args.llm_temperature,
        top_p=args.llm_top_p,
        top_k=args.llm_top_k,
    )
    verifier = LLMVerifier(llm=llm, prompt_name=args.verifier_prompt_name)
    stage = VerifyStage(verifier=verifier)
    stage.run(args.input, args.output, batch_size=args.batch_size)
